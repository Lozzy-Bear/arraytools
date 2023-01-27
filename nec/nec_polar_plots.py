import matplotlib.pyplot as plt
import numpy as np
import pretty_plots


def read_nec(file):
    theta = []
    phi = []
    vertical_power = []
    horizontal_power = []
    total_power = []
    axial_ratio = []
    tilt = []
    sense = []
    volt_theta_mag = []
    volt_theta_phase = []
    volt_phi_mag = []
    volt_phi_phase = []

    skiprows = 999_999
    with open(file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if "- - - RADIATION PATTERNS - - -" in line:
                skiprows = idx + 4
            if idx > skiprows:
                line = ','.join(line.split())
                line = line.split(',')
                if line[0] == '':
                    break
                theta.append(float(line[0]))
                phi.append(float(line[1]))
                vertical_power.append(float(line[2]))
                horizontal_power.append(float(line[3]))
                total_power.append(float(line[4]))
                axial_ratio.append(float(line[5]))
                tilt.append(float(line[6]))
                sense.append(line[7])
                volt_theta_mag.append(float(line[8]))
                volt_theta_phase.append(float(line[9]))
                volt_phi_mag.append(float(line[10]))
                volt_phi_phase.append(float(line[11]))
        data = np.array([theta, phi, vertical_power, horizontal_power, total_power, axial_ratio, tilt,
                         volt_theta_mag, volt_theta_phase, volt_phi_mag, volt_phi_phase])
    return data, sense


def polar_horizontal_gain_plot(data, heading=0.0, fov=None, rmin=None, rmax=None):
    idx = np.argmax(data[4, :])
    pwi = np.argwhere(data[0, :] == data[0, idx])
    a = np.deg2rad(data[1, pwi] - heading)
    r = data[4, pwi]
    if rmin is None:
        rmin = np.min(r)
    if rmax is None:
        rmax = np.max(r)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(a, r)
    ax.plot(np.deg2rad([90.0 - heading, 90.0 - heading]), [rmin, rmax], '--k')
    ax.text(np.deg2rad(90.0 - heading), rmax + 2.0, f'{heading}°')
    # Plot the fov shaded area
    if fov is not None:
        ax.plot(np.deg2rad([90.0-heading-fov, 90.0-heading-fov]), [rmin, rmax], '--r')
        ax.plot(np.deg2rad([90.0-heading+fov, 90.0-heading+fov]), [rmin, rmax], '--r')
        ax.fill_between(np.deg2rad(np.linspace(90.0-heading-fov, 90.0-heading+fov, int(2*fov))), rmin, rmax, alpha=0.1, color='r')
    ax.set_xticks(np.pi / 180. * np.linspace(0, 360, 8, endpoint=False), ['E', '', 'N', '', 'W', '', 'S', ''])
    ax.set_rmin(rmin)
    ax.set_rmax(rmax)
    ax.set_rlabel_position(-22.5)
    ax.grid(True)

    return


def polar_vertical_gain_plot(data, fov=None, rmin=None, rmax=None):
    idx = np.argmax(data[4, :])
    elmax = 90.0 - data[0, idx]
    # print(f'max gain direction: {90.0 - data[0, idx]} [deg] el, {90.0 - data[1, idx]} [deg] az, {data[4, idx]} dB')
    pwi = np.argwhere(data[1, :] == 90.0)
    pwj = np.argwhere(data[1, :] == data[1, idx] + 180.0)
    a = np.deg2rad(90.0 - data[0, pwi])
    a = np.append(a, np.deg2rad(90.0 + data[0, pwj])[::-1])
    r = data[4, pwi]
    r = np.append(r, data[4, pwj][::-1])
    if rmin is None:
        rmin = np.min(r)
    if rmax is None:
        rmax = np.max(r)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(a, r)
    ax.plot(np.deg2rad([elmax, elmax]), [rmin, rmax], '--k')
    ax.text(np.deg2rad(elmax), rmax + 2.0, f'{elmax}°')
    if fov is not None:
        ax.plot(np.deg2rad([0.0, 0.0]), [rmin, rmax], '--r')
        ax.plot(np.deg2rad([fov, fov]), [rmin, rmax], '--r')
        ax.fill_between(np.deg2rad(np.linspace(0.0, fov, int(fov))), rmin, rmax, alpha=0.1, color='r')
    ax.set_xticks(np.pi / 180. * np.linspace(0, 360, 8, endpoint=False),
                  ['          Horizon', '', 'Zenith', '', 'Backlobe          ', '', 'Ground', ''])
    ax.set_rmin(rmin)
    ax.set_rmax(rmax)
    ax.set_thetamax(180.0)
    ax.set_thetamin(0.0)
    ax.grid(True)

    return


def polar_gain_plot(data, heading=0.0, fov=None, rmin=None, rmax=None):
    idx = np.argmax(data[4, :])
    elmax = 90.0 - data[0, idx]
    pwi = np.argwhere(data[0, :] == data[0, idx])
    horz_angle = np.deg2rad(data[1, pwi] - heading)
    horz_power = data[4, pwi]
    pwj = np.argwhere(data[1, :] == 90.0)
    pwk = np.argwhere(data[1, :] == data[1, idx] + 180.0)
    vert_angle = np.deg2rad(90.0 - data[0, pwj])
    vert_angle = np.append(vert_angle, np.deg2rad(90.0 + data[0, pwk])[::-1])
    vert_power = data[4, pwj]
    vert_power = np.append(vert_power, data[4, pwk][::-1])
    if rmin is None:
        rmin = np.min(horz_power)
    if rmax is None:
        rmax = np.max(horz_power)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, subplot_kw={'projection': 'polar'},
                                   figsize=(6, 8), gridspec_kw={'hspace': 0.0, 'bottom': 0.0})
    plt.tight_layout()

    # Horizontal view plot
    ax1.plot(horz_angle, horz_power)
    ax1.plot(np.deg2rad([90.0 - heading, 90.0 - heading]), [rmin, rmax], '--k')
    ax1.text(np.deg2rad(90.0 - heading), rmax + 2.0, f'{heading}°')
    # Plot the fov shaded area
    if fov is not None:
        ax1.plot(np.deg2rad([90.0-heading-fov, 90.0-heading-fov]), [rmin, rmax], '--r')
        ax1.plot(np.deg2rad([90.0-heading+fov, 90.0-heading+fov]), [rmin, rmax], '--r')
        ax1.fill_between(np.deg2rad(np.linspace(90.0-heading-fov, 90.0-heading+fov, int(2*fov))), rmin, rmax, alpha=0.1, color='r')
    ax1.set_xticks(np.pi / 180. * np.linspace(0, 360, 8, endpoint=False), ['E', '', 'N', '', 'W', '', 'S', ''])
    ax1.set_rmin(rmin)
    ax1.set_rmax(rmax)
    ax1.set_rlabel_position(-22.5)
    ax1.grid(True)

    # Vertical view plot
    ax2.plot(vert_angle, vert_power)
    ax2.plot(np.deg2rad([elmax, elmax]), [rmin, rmax], '--k')
    ax2.text(np.deg2rad(elmax), rmax + 2.0, f'{elmax}°')
    # Plot the fov shaded area
    if fov is not None:
        ax2.plot(np.deg2rad([0.0, 0.0]), [rmin, rmax], '--r')
        ax2.plot(np.deg2rad([fov, fov]), [rmin, rmax], '--r')
        ax2.fill_between(np.deg2rad(np.linspace(0.0, fov, int(fov))), rmin, rmax, alpha=0.1, color='r')
    ax2.set_xticks(np.pi / 180. * np.linspace(0, 360, 8, endpoint=False), ['          Horizon', '', 'Zenith', '', 'Backlobe          ', '', 'Ground', ''])
    ax2.set_rmin(rmin)
    ax2.set_rmax(rmax)
    ax2.set_thetamax(180.0)
    ax2.set_thetamin(0.0)
    ax2.grid(True)

    return


def main():
    data, sense = read_nec('ibtx3l.out')
    polar_gain_plot(data, heading=16.0, rmin=-30.0, rmax=30.0)
    # polar_horizontal_gain_plot(data, heading=16.0, fov=45.0, rmin=-30.0, rmax=30.0)
    # polar_vertical_gain_plot(data, fov=45.0, rmin=-30.0, rmax=30.0)
    plt.show()
    # plt.savefig("ib_tx_gain.pdf")
    return


if __name__ == '__main__':
    main()

