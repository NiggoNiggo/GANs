import os
import time
import datetime



class TrainingLogger():
    def __init__(self,name):
        self.name = name
        self.start_time = time.time()



    def add_content(self,text:str):
        """add_content adds content to the logger text file, especially the information string from the command line
        is saved in the logger file

        Parameters
        ----------
        text : str
            per Epoch the information about loss and Scores
        """
        #current date 
        cur_date = datetime.datetime.now()
        # time_elapsed sofar
        elapdes_time = time.time() - self.start_time
        #create the string
        dynamic_string = f"{text} | Date: {cur_date} | Elapsed Time: {elapdes_time:.2f} s"
        #write in the file
        with open(self.name, "a") as file:
            file.write(f"{dynamic_string}\n")




if __name__ == "__main__":
    logging = TrainingLogger("testen")
    print(logging.name)
