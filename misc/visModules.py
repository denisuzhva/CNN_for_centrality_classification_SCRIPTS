import numpy as np
import matplotlib.pyplot as plt



def randomChoose(cen, labels):
    lab_0 = np.argwhere(labels == 0).flatten()
    lab_1 = np.argwhere(labels == 1).flatten()

    ev_class = 0
    n_runs = 10
    for run in range(n_runs):
        if ev_class:
            ev_number = lab_1[np.random.randint(lab_1.size)]
        else:
            ev_number = lab_0[np.random.randint(lab_0.size)]
        event_section = cen[ev_number, :, :, :]

        fig, ax = plt.subplots(4)
        fig.set_figheight(9)
        fig.set_figwidth(6)
        for i in range(4):
            pcm = ax[i].matshow(event_section[i, :, :], cmap='inferno')
        fig.suptitle('class {}, event #{}'.format(ev_class, ev_number))
        fig.colorbar(pcm, ax=ax[:])
        fig.savefig('../EVENT_SECTIONS/{}_{}.png'.format(ev_class, ev_number))
        #plt.show()


def matchEvents(cen, labels_nrg, labels_spec):
    match_events = labels_nrg == labels_spec
    matching = False
    not_match_args = np.argwhere(match_events == matching)

    n_runs = 10
    for run in range(n_runs):
        ev_number = not_match_args[np.random.randint(not_match_args.size)][0]
        event_section = cen[ev_number]

        fig, ax = plt.subplots(4)
        fig.set_figheight(9)
        fig.set_figwidth(6)
        for i in range(4):
            pcm = ax[i].matshow(event_section[i, :, :], cmap='inferno')
        fig.suptitle(f'event #{ev_number}, matching: {matching}')
        fig.colorbar(pcm, ax=ax[:])
        fig.savefig(f'../EVENT_SECTIONS/{matching}_{ev_number}.png')
        #plt.show()


if __name__ == '__main__':
    DATASET = '../../DATASETS/NA61_EPOS_400prof/'

    cen = np.load(DATASET + 'features_central.npy').reshape((-1, 4, 4, 10))
    labels_nrg = np.load(DATASET + 'labels_nrg.npy').argmax(axis=1)
    labels_spec = np.load(DATASET + 'labels_spec.npy').argmax(axis=1)

    matchEvents(cen, labels_nrg, labels_spec)