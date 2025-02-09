from AdvEx_RL_MuJoCo.test_run import run as advexrl_run
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configure-env', default='none', help='Set test environment to setup all configuration')
    parser.add_argument('--exp-data-dir', default='/Experimental_Data/', help='Set experiment data location')
    arg = parser.parse_args()
    name = arg.configure_env
    test_epi_no = 100
    advexrl_run(env_name=name, eval_epi_no=test_epi_no, exp_data_dir=arg.exp_data_dir)
