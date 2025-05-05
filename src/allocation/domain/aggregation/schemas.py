from pydantic import RootModel
from typing import Dict

class CombinedImageResponse(RootModel[Dict[str, str]]):
    pass