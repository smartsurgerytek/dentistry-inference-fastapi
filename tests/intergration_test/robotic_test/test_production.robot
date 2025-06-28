*** Settings ***
Library           RequestsLibrary
Library           OperatingSystem
Library           Collections
Library           BuiltIn

Suite Setup       Create Session To API

*** Variables ***
${API_DOMAIN_URL}      https://api.smartsurgerytek.net/dentistry-stg
${IMG1_B64_PATH}       ./tests/files/PA_b64.txt    # PA 圖 (文字檔)
${IMG2_B64_PATH}       ./tests/files/pano_b64.txt    # Pano 圖 (文字檔)
${IMG3_B64_PATH}       ./tests/files/black_b64.txt    # 黑圖 (文字檔)

*** Keywords ***
Create Session To API
    ${yaml_text}=    Get File    ./conf/credential.yaml
    ${yaml}=         Evaluate    __import__('yaml').safe_load('''${yaml_text}''')
    ${apikey}=       Set Variable    ${yaml['DENTISTRY_API_KEY']}
    Run Keyword If   '${apikey}' == 'please write the token here'    Fail    Please set DENTISTRY_API_KEY in credential.yaml or as an env variable.
    Set Suite Variable    ${API_KEY}    ${apikey}
    ${headers}=      Create Dictionary    Authorization=Bearer ${API_KEY}
    Create Session   api    ${API_DOMAIN_URL}    headers=${headers}    verify=False

Read Base64 From Text File
    [Arguments]    ${path}
    ${b64}=    Get File    ${path}
    RETURN    ${b64}

Send Inference Request
    [Arguments]    ${path}    ${image_base64}
    &{params}=    Create Dictionary    apikey=${API_KEY}
    &{json}=      Create Dictionary    image=${image_base64}
    ${response}=  Post Request    api    ${path}    params=${params}    json=${json}
    Log    Path: ${path}
    Log    Status: ${response.status_code}
    Should Be Equal As Integers    ${response.status_code}    200

Get All Paths
    ${paths}=    Create List
    Append To List    ${paths}    /pa/root-segmentation
    Append To List    ${paths}    /pa/caries-detection
    Append To List    ${paths}    /pano/fdi-segmentation
    RETURN    ${paths}

*** Test Cases ***
Test All Endpoints With PA and Pano Images From Base64 Txt
    ${img1_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    ${img2_b64}=    Read Base64 From Text File    ${IMG2_B64_PATH}
    ${all_paths}=   Get All Paths

    FOR    ${path}    IN    @{all_paths}
        Run Keyword If    'pa' in '${path}'    Send Inference Request    ${path}    ${img1_b64}
        ...    ELSE IF    'pano' in '${path}'    Send Inference Request    ${path}    ${img2_b64}
        ...    ELSE    Fail    Unknown path pattern: ${path}
    END

Test All Endpoints With Black Image From Base64 Txt
    ${img3_b64}=    Read Base64 From Text File    ${IMG3_B64_PATH}
    ${all_paths}=   Get All Paths

    FOR    ${path}    IN    @{all_paths}
        Send Inference Request    ${path}    ${img3_b64}
    END
