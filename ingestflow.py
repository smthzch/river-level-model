from metaflow import FlowSpec, step, catch
from riverlevelcam import ingest

class DataFlow(FlowSpec):
    """A flow to retrieve river camera imagery and associated stage/discharge measurements.

    The flow start with getting stage data for images that already exist and storing it within the DB

    It will then retrieve any new imagery.
    
    """
    
    @step
    def start(self):
        ingest.createDB()
        self.next(self.check_images, self.usgs_data)

    @step
    def check_images(self):
        self.imgs_to_get = ingest.imgsINdb()
        self.next(self.match_data)

    @step
    def usgs_data(self):
        self.data = ingest.usgsData()
        self.next(self.match_data)

    @step
    def match_data(self, inputs):
        imgs = inputs.check_images.imgs_to_get
        data = inputs.usgs_data.data

        self.augdata = ingest.matchData(imgs, data)
        self.next(self.insert_data)

    @step
    def insert_data(self):
        ingest.insertData(self.augdata)
        self.next(self.get_imagery)

    @catch(var="get_imagery_failed")
    @step
    def get_imagery(self):
        ingest.save_image_as_date()
        self.next(self.end)

    @step
    def end(self):
        print("Done.")

if __name__ == "__main__":
    DataFlow()