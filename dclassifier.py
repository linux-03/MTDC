import pandas as pd
import numpy as np

class DC_EVENT_DETECTOR:
    
    ## comment for trading strategy: if you want to know if a dc event has finished to classify and regress its os length have a look at the length od list returned by get_dc
    def __init__(self, theta: float, start_time: int, start_price: float) -> None:
        self.p_h = start_price
        self.p_l = start_price
        self.dc_end = start_time
        self.dc_start = start_time
        self.os_start = start_time
        self.os_end = start_time
        self.event_up = True
        self.theta = theta
        self.os_events = []
        self.dc_events = []
    

    def step(self, price, time):
        if self.event_up:
            #while being in upturning os event but still looking for directional change downwards
            if price <= self.p_h*(1-self.theta):
                self.event_up = False # we now have a downturn os event
                self.p_l = price
                self.dc_end = time # set end time of dc downturn event
                self.os_events.append([self.os_start, self.os_end])
                self.classifie_dc(trend='Down')
                self.os_start = time + 1
                self.os_end = time + 1 
                self.dc_start = time + 1
            
            else:
                # upgoing os event - set new highest price
                if self.p_h < price:
                    self.p_h = price
                    self.dc_start = time # we set dc downturn start point if we never return back higher than now
                    self.os_end = time - 1
        
        else:
            #while being in downturning os event but still looking for directional change upwards
            if price >= self.p_l*(1+self.theta):
                self.event_up = True # we now have a downtrun os event
                self.p_h = price
                self.dc_end = time #set end time of upturn os event
                self.os_events.append([self.os_start, self.os_end])
                self.classifie_dc(trend = 'Up')
                self.dc_start = time + 1
                self.os_end = time + 1
                self.os_start = time + 1

            
            else:
                # downgoing os event - set new lowest price
                if self.p_l > price:
                    self.p_l = price
                    self.dc_start = time # we set dc upturn start point if we never return back lower than now
                    self.os_end = time - 1
    
    def classifie_dc(self, trend = 'Up'):
        dc_event = {"start": self.dc_start,
                 "end": self.dc_end,
                 "event_time": self.dc_end - self.dc_start,
                 "event_price": (self.p_h - self.p_l) if trend=='Up' else (self.p_l - self.p_h),
                 "prev_dc_end_price": 0,
                 "prev_os": False if self.os_end == self.os_start else True,
                 "flash": True if self.dc_end == self.dc_start else False}
        
        
        if len(self.dc_events) == 0:
            dc_event['prev_os'] = True
            dc_event['prev_dc_end_price'] = self.p_l if trend=='Up' else self.p_h
        else:
            dc_event['prev_os'] = self.dc_events[-1]['end']
        

        self.dc_events.append(dc_event)

    def get_os(self):
        return self.os_events
    

    def get_dc(self):
        return self.dc_events
    
    def get_dc_data(self):
        return pd.DataFrame(self.dc_events_data[-1])
    
    def get_dc_count(self):
        return len(self.dc_events)




def classify_split_timeseries(data: pd.DataFrame, theta):
    dc_handler = DC_EVENT_DETECTOR(theta, 0, data.iloc[0])
    dc_class = []

    for i in range(len(data)):
        dc_handler.step(data.iloc[i], i)

    dc_events = dc_handler.get_dc()

    prev_end = dc_events[0]['end']
    for el in range(1, dc_handler.get_dc_count()):
        if not (dc_events[el]['start'] - prev_end):
            dc_class.append(1)
        else:
            dc_class.append(0)

    dc_events = dc_events[:-1] #delete last DC event as we dont know its type

    return pd.DataFrame(dc_events), dc_class


class DC_EVENT_HANDLER:

    def __init__(self, theta, classifier, regressor) -> None:
        self.theta = theta
        self.classifier = classifier
        self.regressor = regressor
        self.data = pd.DataFrame
        self.detector = None
        self.dc_count = 0
        
    

    def set_new_data(self, data):
        self.data = data
        self.detector = DC_EVENT_DETECTOR(self.theta, 0, data['Price'].iloc[0])
        self.dc_count = 0
    
    def step(self, price, time):
        self.detector.step(price, time)
        
        # if no new dc_event happend => return 0
        if self.dc_count == self.detector.get_dc_count():
            return 0

        # if new dc_event happend

        dc_data = self.detector.get_dc_data()
        dc_class = self.classifier.predict(dc_data) # 1 means alpha event 0 means not alpha

        # if dc event is alpha => return
        if dc_class:
            return 0
        
        return self.regressor.predit(dc_class['event_time']), 'Up' if dc_data['event_price'] > 0 else 'Down'


