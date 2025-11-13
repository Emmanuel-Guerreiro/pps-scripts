from roboflow import Roboflow
rf = Roboflow(api_key="ooPxxzLT1xgs8CWEiMjY")
project = rf.workspace("clment-le-padellec").project("luggage-cuaxr")
version = project.version(1)
dataset = version.download("yolov11")
                