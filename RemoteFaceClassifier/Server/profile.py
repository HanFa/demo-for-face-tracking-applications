import shutil, time, cProfile, StringIO, pstats
from enum import Enum
from RemoteFaceClassifier.Server import *
import RemoteFaceClassifier.Server.globals as globals

MEASURE_TYPE = Enum('MEASURE_TYPE', 'TRANSMISSION, LOCATE, CLASSIFY, TOTAL')

class ServerProfiler:
    """ Performance profiler for face tracking server.

    For each frame, it keeps track of:
        1. transmission time from client to server side
        2. time to locate all face boundaries
        3. time to classify in torch subprocess
    """

    def append_log_line(self, info):
        with open(self.profile, 'a') as f:
            f.write(info)

    def __init__(self, enable, profile_dir):
        self.enable = enable
        self.profile_dir = profile_dir
        self.profile = os.path.join(profile_dir, 'measure.csv')
        self.cprofile_prefix = os.path.join(profile_dir, 'cprofile_measure_frame_')

        self.transmission_start_time = time.time()
        self.locate_start_time = time.time()
        self.classify_start_time = time.time()
        self.total_start_time = time.time()

        self.transmission_time = 0.0
        self.locate_time = 0.0
        self.classify_time = 0.0
        self.total_time = 0.0

        self.cprofiler = cProfile.Profile()
        self.cprofiler_io = StringIO.StringIO()

        if os.path.exists(profile_dir):
            shutil.rmtree(profile_dir)

        os.mkdir(profile_dir)
        self.append_log_line('frame_num\ttransmit\tlocate\tclassify\ttotal\n')


    def inform_transmission_time_start(self, measure_type):
        if not self.enable:
            return

        if measure_type == MEASURE_TYPE.TRANSMISSION:
            self.transmission_start_time = time.time()
        elif measure_type == MEASURE_TYPE.LOCATE:
            self.locate_start_time = time.time()
        elif measure_type == MEASURE_TYPE.CLASSIFY:
            self.classify_start_time = time.time()
        elif measure_type == MEASURE_TYPE.TOTAL:
            self.total_start_time = time.time()
            self.cprofiler.enable()

        return


    def inform_transmission_time_stop(self, measure_type):
        if not self.enable:
            return

        if measure_type == MEASURE_TYPE.TRANSMISSION:
            self.transmission_time = time.time() - self.transmission_start_time
        elif measure_type == MEASURE_TYPE.LOCATE:
            self.locate_time = time.time() - self.locate_start_time
        elif measure_type == MEASURE_TYPE.CLASSIFY:
            self.classify_time = time.time() - self.classify_start_time
        elif measure_type == MEASURE_TYPE.TOTAL:
            self.total_time = time.time() - self.total_start_time
            self.cprofiler.disable()

        return


    def update_log(self):
        if not self.enable:
            return

        global frame_num
        self.append_log_line(
            '{}\t{}\t{}\t{}\t{}\n'.format(
                globals.frame_num, self.transmission_time, self.locate_time, self.classify_time, self.total_time))

        self.transmission_time = 0.0
        self.locate_time = 0.0
        self.classify_time = 0.0

        self.cprofiler.create_stats()
        self.cprofiler.dump_stats(self.cprofile_prefix + str(globals.frame_num) + '.pyprof')
        return


profiler = ServerProfiler(SERVER_PROFILE_ENABLE, SERVER_PROFILE_DIR) # Profiler for performance measure
