"""
Event Latency example

Covers:

- Resources: Store

Scenario:
  This example shows how to separate the time delay of events between
  processes from the processes themselves.

When Useful:
  When modeling physical things such as cables, RF propagation, etc.  it
  better encapsulation to keep this propagation mechanism outside of the
  sending and receiving processes.

  Can also be used to interconnect processes sending messages

Example by:
  Keith Smith

"""
import simpy
from desmod.queue import Queue

SIM_DURATION = 100


class Cable(object):
    """This class represents the propagation through a cable."""
    def __init__(self, env, delay):
        self.env = env
        self.delay = delay
        self.store = Queue(env, capacity=2, hard_cap=True)

    def latency(self, value):
        # yield self.env.timeout(self.delay)
        self.store.put(value)

    def put(self, value):
        return self.store.put(value)
        # self.env.process(self.latency(value))

    def get(self):
        return self.store.get()


def sender(env, cable):
    """A process which randomly generates messages."""

    yield cable.put(1)
    yield env.timeout(1)
    yield cable.put(1)
    yield env.timeout(1)
    # with raises(OverflowError):
    try:
        put_evt = cable.put(1)
        yield put_evt
    except OverflowError as err:
        print(1234)
    # while True:
    #     # wait for next transmission
    #     yield env.timeout(1)
    #     put_evt = cable.put('Sender sent this at %d' % env.now)
    #     try:
    #         a = yield env.any_of([put_evt, env.timeout(1)])
    #         print(a.events)
    #     except OverflowError as err:
    #         print('123')


def clock(env):
    while True:
        yield env.timeout(1)
        print('clock wake up at', env.now)

def clock_watcher(env, num, clock_proc):
    """A process which consumes messages."""
    while True:
        # Get event for message pipe
        yield clock_proc.target
        # msg = yield cable.get()
        print('watch %d wake up at %d' % (num, env.now))


# Setup and start the simulation
print('Event Latency')
env = simpy.Environment()

cable = Cable(env, 10)
clock_proc = env.process(clock(env))
env.process(clock_watcher(env,1, clock_proc))
env.process(clock_watcher(env,2, clock_proc))


# env.process(receiver(env, cable))
# def start(env):
#     try:
#         a = yield env.any_of( [sender_proc, env.timeout(1)]  )
#         print(a.events)
#     except OverflowError as err:
#         print('123')
# sender_proc = env.process(start(env))

env.run(until=SIM_DURATION)