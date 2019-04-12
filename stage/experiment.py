import json
import argparse

def parsing_arguments():
    parser = argparse.ArgumentParser(description='Arguments for creating Experiment')
    parser.add_argument('--exp',  nargs='+', help='Experiment Name: This should be unique', type=str, default="Exp101")
    parser.add_argument('--mode', type=int, default=0, help='Mode of operation, 0- Episode, 1-1000 steps in one episode')
    parser.add_argument('--data', type=int, default=1, help ='Data generation, 1 -1D and 2- 2D')
    parser.add_argument('--shape', type=int, default=1, help ='Data shape, 1 -Quadratic and 2- Local Minima Data')
    parser.add_argument('--q', type=int, default=5, help ='Queue Length, 1 -1D and 2- 2D')
    parser.add_argument('--q_c', type=int, default=2, help ='Queue lenght used in Heuristic')
    parser.add_argument('--y', type=float, default=0.9, help ='discount factor')
    parser.add_argument('--c', type=int, default=7, help ='Heuristic clip')
    parser.add_argument('--e_start', type=float, default=1, help ='Start applying Heuristic')
    parser.add_argument('--e_stop', type=float, default=0, help ='Stop applying Heuristic')
    parser.add_argument('--epi', type=int, default=300, help ='Number of episodes per experiment')
    parser.add_argument('--l', type=float, default=0.9, help ='learning rate')
    parser.add_argument('--r', type=int, default=1, help='1 - for reward + delayed reward, 2- for reward in the state')
    parser.add_argument('--lam', type=float, default=0.9, help='rate of decay')
    parser.add_argument('--skip', type=int, default=3, help='minimum skip')
    parser.add_argument('--create', type=int, default=0, help ='Create experiment, 1- active , 0 not active')
    parser.add_argument('--view', type=int, default=0, help ='Create experiment, 1- active , 0 not active')

    return parser.parse_args()

def create_experiment(parameters):
        filename = 'experiments/'+parameters['exp']+'.json'
        with open(filename, 'w') as f:
            json.dump(parameters, f )


def view_experiment(name, desc=False):
    filename = 'experiments/'+name+'.json'
    data={}
    with open(filename, 'r') as f:
        data = json.load(f)
    if desc:
        mode = "Episode" if data['mode'] ==0 else "1000 steps in one episode"
        shape = "Quadratic " if data['shape']==1 else "Local Minima"
        reward = "Reward + Discounted Future Reward" if data['r']==1 else "Reward at the state"
    
        des="Experiment Name : {} , Mode : Runing in {} mode, Data is {}D, with  {} Function. Parameters: Learning Rate :{},\
            Discount Factor {}, Reward {}, Epsilon Start {}, Epsilon Stop {}, Queue Length {}, Consecutive Queue Length {}, Decay Rate {} \
            , Clip {} , Minimum Skip {}".format(data['exp'],
                    mode, 
                    data['data'], 
                    shape,
                    data['l'],
                    data['y'],
                    reward,
                    data['e_start'],
                    data['e_stop'],
                    data['q'],
                    data['q_c'],
                    data['lambda'],
                    data['c'],
                    data['skip']
             )  
        print(des)
    return data



def main():
    args = parsing_arguments()
    if args.create!=0:
        print("Create mode activated")
        parameters={
            "exp": args.exp,
            "mode": args.mode,
            "data": args.data,
            "shape": args.shape,
            "q": args.q,
            "q_c": args.q_c,
            "y": args.y,
            "c":args.c,
            "e_start": args.e_start,
            "e_stop":args.e_stop,
            "epi": args.epi,
            "l": args.l,
            "r":args.r,
            "lambda":args.lam,
            "skip": args.skip 
        }

        create_experiment(parameters)

    elif args.view!=0:
        print("View Experiment Parameters")

        data = view_experiment(args.exp)







if __name__ == "__main__":
    main()
