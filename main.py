from get_data import Data
from model import Model
from visual import Visual


class Settings:
    def __init__(self, data, my_model, visible):
        self.data_all = data
        self.model = my_model
        self.visual = visible

    def run(self):
        self.model.train_model()
        self.visual.visual_train()
        self.model.test_model()
        if self.model.name == 'resnet':
            self.visual.visual_test()
        self.visual.draw_matrix()
        if self.model.name == 'resnet':
            self.visual.show_feature()


if __name__ == '__main__':
    data_all = Data
    model = Model(data_all)
    visual = Visual(model)
    settings = Settings(data_all, model, visual)
    settings.run()
