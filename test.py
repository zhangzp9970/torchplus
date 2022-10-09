from torchplus.utils import Init
root_dir = "./log"
init = Init(seed=9970, log_root_dir=root_dir,
            backup_filename=__file__, tensorboard=True, comment=f'test')
output_device = init.get_device()
writer = init.get_writer()
log_dir = init.get_log_dir()
data_workers = init.get_workers()
parser=init.get_parser()
parser.add_argument('--hidden_layer', type=int, default=128)
args = parser.parse_args()
pass
