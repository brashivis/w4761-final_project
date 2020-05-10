import matplotlib.pyplot as plt

def pull_error_from_log(log):
    err_lines = []
    for line in log:
        if line.find('err: ') != -1: err_lines.append(line)

    err_vals = []
    for l in err_lines:
        t = l.split('err: ')[1] 
        t = t.split()[0]
        err_vals.append(float(t)) 

    return err_vals

def plot_error(err_vals, plot_filename):
    plt.plot(err_vals)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(plot_filename)

if __name__ == '__main__':
    filename = 'log.txt'
    log_file = open(filename)
    err_vals = pull_error_from_log(log_file)

