import time

def tic():
    #Homemade version of matlab tic and toc functions
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc(display=""):

    if 'startTime_for_tictoc' in globals():
        print (display +" Elapsed time is "+ str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")
