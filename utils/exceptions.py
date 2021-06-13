

class InsufficientDpException(RuntimeError):
    # remaining dp in a block is not enough for a task commit
    pass


class RejectDpPermissionError(RuntimeError):
    pass


class StopReleaseDpError(RuntimeError):
    # time based release run out of DP
    pass


class DpBlockRetiredError(RuntimeError):
    pass


class DprequestTimeoutError(RuntimeError):
    pass


class ResourceAllocFail(RuntimeError):
    pass


class TaskPreemptedError(RuntimeError):
    pass


class RejectResourcePermissionError(RuntimeError):
    pass
