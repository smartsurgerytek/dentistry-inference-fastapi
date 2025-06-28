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
    ${yaml}=    Evaluate    __import__('yaml').safe_load('''${yaml_text}''')
    ${apikey}=    Set Variable    ${yaml['DENTISTRY_API_KEY']}
    Run Keyword If    '${apikey}' == 'please write the token here'    Fail    Please write the token in credential.yaml
    Set Suite Variable    ${API_KEY}    ${apikey}
    ${headers}=    Create Dictionary    Authorization=Bearer ${API_KEY}
    Create Session    api    ${API_DOMAIN_URL}    headers=${headers}

Read Base64 From Text File
    [Arguments]    ${path}
    ${b64}=    Get File    ${path}
    RETURN    ${b64}

Send Inference Request
    [Arguments]    ${path}    ${image_base64}    ${expected_status}=200
    &{params}=    Create Dictionary    apikey=${API_KEY}
    &{json}=      Create Dictionary    image=${image_base64}
    ${result}=    Run Keyword And Ignore Error    POST On Session    api    ${path}    params=${params}    json=${json}
    ${status}=    Set Variable    400
    IF    '${result[0]}' == 'PASS'
        Set Test Variable    ${status}    ${result[1].status_code}
    ELSE
        Log    Request failed: ${result[1]}
    END
    Should Be Equal As Integers    ${status}    ${expected_status}

*** Test Cases ***

Test PA Aggregation Images - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pa_aggregation_images    ${img_b64}    200

Test PA Aggregation Images - Empty
    Send Inference Request    /v1/pa_aggregation_images    ""    400

Test PA Aggregation Images - Malformed
    Send Inference Request    /v1/pa_aggregation_images    "%%%@@!!invalid"    400

Test PA Measure CVAT - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pa_measure_cvat    ${img_b64}    200

Test PA Measure CVAT - Empty
    Send Inference Request    /v1/pa_measure_cvat    ""    400

Test PA Measure CVAT - Malformed
    Send Inference Request    /v1/pa_measure_cvat    "%%%@@!!invalid"    400

Test PA Measure Dict - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pa_measure_dict    ${img_b64}    200

Test PA Measure Dict - Empty
    Send Inference Request    /v1/pa_measure_dict    ""    400

Test PA Measure Dict - Malformed
    Send Inference Request    /v1/pa_measure_dict    "%%%@@!!invalid"    400

Test PA Measure Image - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pa_measure_image    ${img_b64}    200

Test PA Measure Image - Empty
    Send Inference Request    /v1/pa_measure_image    ""    400

Test PA Measure Image - Malformed
    Send Inference Request    /v1/pa_measure_image    "%%%@@!!invalid"    400

Test PA Pano Classification Dict - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pa_pano_classification_dict    ${img_b64}    200

Test PA Pano Classification Dict - Empty
    Send Inference Request    /v1/pa_pano_classification_dict    ""    400

Test PA Pano Classification Dict - Malformed
    Send Inference Request    /v1/pa_pano_classification_dict    "%%%@@!!invalid"    400

Test PA Segmentation CVAT - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pa_segmentation_cvat    ${img_b64}    200

Test PA Segmentation CVAT - Empty
    Send Inference Request    /v1/pa_segmentation_cvat    ""    400

Test PA Segmentation CVAT - Malformed
    Send Inference Request    /v1/pa_segmentation_cvat    "%%%@@!!invalid"    400

Test PA Segmentation Image - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pa_segmentation_image    ${img_b64}    200

Test PA Segmentation Image - Empty
    Send Inference Request    /v1/pa_segmentation_image    ""    400

Test PA Segmentation Image - Malformed
    Send Inference Request    /v1/pa_segmentation_image    "%%%@@!!invalid"    400

Test PA Segmentation YOLOv8 - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pa_segmentation_yolov8    ${img_b64}    200

Test PA Segmentation YOLOv8 - Empty
    Send Inference Request    /v1/pa_segmentation_yolov8    ""    400

Test PA Segmentation YOLOv8 - Malformed
    Send Inference Request    /v1/pa_segmentation_yolov8    "%%%@@!!invalid"    400

Test Pano Caries Detection Dict - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pano_caries_detection_dict    ${img_b64}    200

Test Pano Caries Detection Dict - Empty
    Send Inference Request    /v1/pano_caries_detection_dict    ""    400

Test Pano Caries Detection Dict - Malformed
    Send Inference Request    /v1/pano_caries_detection_dict    "%%%@@!!invalid"    400

Test Pano Caries Detection Image - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pano_caries_detection_image    ${img_b64}    200

Test Pano Caries Detection Image - Empty
    Send Inference Request    /v1/pano_caries_detection_image    ""    400

Test Pano Caries Detection Image - Malformed
    Send Inference Request    /v1/pano_caries_detection_image    "%%%@@!!invalid"    400

Test Pano FDI Segmentation CVAT - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pano_fdi_segmentation_cvat    ${img_b64}    200

Test Pano FDI Segmentation CVAT - Empty
    Send Inference Request    /v1/pano_fdi_segmentation_cvat    ""    400

Test Pano FDI Segmentation CVAT - Malformed
    Send Inference Request    /v1/pano_fdi_segmentation_cvat    "%%%@@!!invalid"    400

Test Pano FDI Segmentation Image - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pano_fdi_segmentation_image    ${img_b64}    200

Test Pano FDI Segmentation Image - Empty
    Send Inference Request    /v1/pano_fdi_segmentation_image    ""    400

Test Pano FDI Segmentation Image - Malformed
    Send Inference Request    /v1/pano_fdi_segmentation_image    "%%%@@!!invalid"    400

Test Pano FDI Segmentation YOLOv8 - Valid
    ${img_b64}=    Read Base64 From Text File    ${IMG1_B64_PATH}
    Send Inference Request    /v1/pano_fdi_segmentation_yolov8    ${img_b64}    200

Test Pano FDI Segmentation YOLOv8 - Empty
    Send Inference Request    /v1/pano_fdi_segmentation_yolov8    ""    400

Test Pano FDI Segmentation YOLOv8 - Malformed
    Send Inference Request    /v1/pano_fdi_segmentation_yolov8    "%%%@@!!invalid"    400

