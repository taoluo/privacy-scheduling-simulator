from desmod.queue import Queue, FilterQueue, QueueGetEvent
from desmod.pool import Pool
from simpy.events import PENDING, EventPriority, URGENT

class NoneBlockingPutMixin(object):
    def put(self, *args, **kwargs):
        event = super(NoneBlockingPutMixin, self).put(*args, **kwargs)
        assert event.ok
        return event

class SingleGetterMixin(object):
    def get(self, *args, **kwargs):
        event = super(SingleGetterMixin, self).get(*args, **kwargs)
        assert len(self._get_waiters) <= 1
        return event


class NoneBlockingGetMixin(object):
    def get(self, *args, **kwargs):
        event = super(NoneBlockingGetMixin, self).get(*args, **kwargs)
        assert event.ok
        return event
    

class LazyAnyFilterQueue(FilterQueue):
    LAZY: EventPriority = EventPriority(99)

    def _trigger_when_at_least(self, *args, **kwargs) -> None:
        super()._trigger_when_at_least(priority=self.LAZY, *args, **kwargs)


class DummyPutPool(SingleGetterMixin, NoneBlockingPutMixin, Pool):
    pass


class DummyPool(NoneBlockingGetMixin, NoneBlockingPutMixin, Pool):
    pass


class DummyPutQueue(SingleGetterMixin, NoneBlockingPutMixin, Queue):
    pass


class DummyPutLazyAnyFilterQueue(
    SingleGetterMixin, NoneBlockingPutMixin, LazyAnyFilterQueue
):
    pass


class DummyQueue(NoneBlockingPutMixin, NoneBlockingGetMixin, Queue):
    pass


class DummyFilterQueue(NoneBlockingPutMixin, NoneBlockingGetMixin, FilterQueue):
    pass
