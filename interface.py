class PotholeEvent():
    ''' Interface for event that identify a pothole, with attached data.

    Parameters:
    start_at = initial timestamp of detected pothole
    end_at = ending timestamp of detected pothole
    latitude = gps latitude assigned to pothole
    longitude = gps latitude assigned to pothole
    attached_sensors_data = List[SensorData] occurred during PotholeEvent
    attached_images = List[string] path or uid of images (to be defined) captured during PotholeEvent
    '''

    def __init__(self, start_at: int=None, end_at: int=None):
        super().__init__()
        self.bumpID = None
        self.start_at = start_at
        self.end_at = end_at
        self.latitude = None
        self.longitude = None
        self.attached_sensors_data = []
        self.attached_images = []

    def from_dict(self, dict):
        for attr in dict.keys():
            if hasattr(self, attr):
                setattr(self, attr, dict[attr])
            else:
                print("{} is not a class attribute".format(attr))
        return self

    def to_dict(self):
        d = {}
        attrs = ["start_at", "end_at", "latitude", "longitude", "attached_sensors_data", "attached_images"]
        for attr in attrs:
            if hasattr(self, attr):
                d[attr] = getattr(self, attr)
            else:
                d[attr] = None
        return d


    def contain_event(self, start_at: int, end_at: int) -> bool:
        ''' Contain timestamp defined by start_at, end_at (less or equal). '''
        return self.start_at <= start_at and end_at <= self.end_at

    def overlap_event(self, start_at: int, end_at: int) -> bool:
        ''' Overlap timestamp defined by start_at, end_at (less or equal). '''
        overlap_left = self.start_at >= start_at and start_at <= self.end_at
        overlap_left = self.start_at <= end_at and end_at <= self.end_at
        contain_in = self.contain_event(start_at, end_at)
        return overlap_left or overlap_left or contain_in
