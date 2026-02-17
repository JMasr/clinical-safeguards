from enum import Enum


class Label(str, Enum):
    VALID = "Valid"
    MALIGN = "Malign"
    CRISIS = "Crisis"
    ERROR = "Server Error"


class ResponseCode(int, Enum):
    VALID = 100
    MALIGN = 400
    CRISIS = 406
    ERROR = 500


# Precedence used by the pipeline merger: higher index = higher priority.
LABEL_PRECEDENCE: dict[Label, int] = {
    Label.VALID: 0,
    Label.ERROR: 1,
    Label.MALIGN: 2,
    Label.CRISIS: 3,
}

LABEL_TO_CODE: dict[Label, ResponseCode] = {
    Label.VALID: ResponseCode.VALID,
    Label.MALIGN: ResponseCode.MALIGN,
    Label.CRISIS: ResponseCode.CRISIS,
    Label.ERROR: ResponseCode.ERROR,
}
