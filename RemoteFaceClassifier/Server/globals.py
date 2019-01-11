from threading import Condition

frame_num = 0 # Number of frames that server has handled
align_dir_cv = Condition() # Condition variable for Stateful Additional Dataset
