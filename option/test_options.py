from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--checkpoint_name', default=' ', type=str)
        self.parser.add_argument('--results_dir', type=str, default=' ', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='val, test, etc')
        self.parser.add_argument('--initialize', type=bool, default=False, help='')

        self.isTrain = False
