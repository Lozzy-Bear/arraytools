import matplotlib.pyplot as plt
import numpy as np
import pretty_plots
import pymap3d as pm
import os


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


if __name__ == '__main__':
    tx_spacing = 3
    tx_rotation = 16.0

    path = os.path.dirname(__file__)
    path = path.replace("\\", "/")
    tx_data, _ = read_nec(f'{path}/models/ibtx{tx_spacing}l.out')
    rx_data, _ = read_nec(f'{path}/models/ibrx.out')
    y = np.arange(52, 62 + 0.1, 0.1)  # lats
    x = np.arange(-112, -98 + 0.1, 0.1)  # lons
    lons, lats = np.meshgrid(x, y)
    tx_lat = 50.893
    tx_lon = -109.403
    rx_lat = 52.243
    rx_lon = -106.450

    alts = np.arange(70_000, 131_000, 1_000)
    # alts = np.asarray([110_000])
    gm = np.zeros((y.shape[0], x.shape[0], alts.shape[0], 1))

    for k, alt in enumerate(alts):
        print(alt, alt/1000)
        m = np.zeros(lons.shape)
        print('mshape', m.shape)

        for i in range(lons.shape[0]):
            for j in range(lats.shape[1]):
                az1, el1, r1 = pm.geodetic2aer(lats[i, j], lons[i, j], alt,
                                               tx_lat, tx_lon, 0.0, ell=pm.Ellipsoid("wgs84"), deg=True)
                az2, el2, r2 = pm.geodetic2aer(lats[i, j], lons[i, j], alt,
                                               rx_lat, rx_lon, 0.0, ell=pm.Ellipsoid("wgs84"), deg=True)
                az1 = np.round(((90.0 + tx_rotation - az1) % 360.0))
                az2 = np.round(((90.0 + 7.0 - az2) % 360.0))
                el1 = np.round(90.0 - el1)
                el2 = np.round(90.0 - el2)
                pw1 = np.argwhere((tx_data[0, :] == el1) & (tx_data[1, :] == az1))
                pw2 = np.argwhere((rx_data[0, :] == el2) & (rx_data[1, :] == az2))
                pw = tx_data[4, pw1] + rx_data[4, pw2]
                if pw.size == 0:
                    pw = np.nan
                else:
                    pw = pw.flatten()[0]
                m[i, j] = pw
                gm[i, j, k, 0] = pw

        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6.4, 4.8])
        # plt.title(f"Altitude = {alt/1000} km")
        # ax.set_xlabel("Longitude [deg]")
        # ax.set_ylabel("Latitude [deg]")
        # im = ax.imshow(m, origin='lower', extent=[0, 140, 0, 100],
        #                cmap='inferno', vmin=0.0, vmax=30.0, interpolation='bicubic')
        # plt.colorbar(im, label='Link Gain [dB]', shrink=0.72)
        # ax.set_xticks(np.arange(0, 140 + 20, 20), labels=np.arange(-112, -98 + 2, 2))
        # ax.set_yticks(np.arange(0, 100 + 20, 20), labels=np.arange(52, 62 + 2, 2))
        # ax.grid(which='minor', color='Grey', linestyle=':', linewidth=0.5)
        # ax.grid(which='major', color='Grey', linestyle=':', linewidth=0.5)
        # ax.minorticks_on()
        # plt.tight_layout()
        # # plt.savefig(f"ib_tx_gain_3l_{int(alt/1000)}.pdf")
        # plt.savefig(f"ib_link_gain_3l_{int(alt/1000)}.pdf")
        # plt.show()

        # 3D Map
        # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        # u, v = np.mgrid[0:np.pi:360j, 0:2 * np.pi:360j]
        # r = 1.0
        # x = r * np.sin(u) * np.cos(v)
        # y = r * np.sin(u) * np.sin(v)
        # z = r * np.cos(u)
        # rr = 1.01
        # xx = rr * np.sin(np.deg2rad(lats)) * np.cos(np.deg2rad(lons))
        # yy = rr * np.sin(np.deg2rad(lats)) * np.sin(np.deg2rad(lons))
        # zz = rr * np.cos(np.deg2rad(lats))
        # print(x.shape, y.shape, z.shape)
        # # ax.plot_surface(x, y, z)
        # scamap = plt.cm.ScalarMappable(cmap='inferno')
        # fcolors = scamap.to_rgba(np.nan_to_num(np.flipud(m), nan=-40))
        # ax.plot_surface(xx, yy, zz, facecolors=fcolors, cmap='inferno')

        # plt.show()

    import h5py
    hf = h5py.File(f'ib3d_link_gain_mask_{tx_spacing}lam_rot{tx_rotation}.h5', 'w')
    hf.create_dataset('gain_mask', data=gm)
    hf.create_dataset('latitude', data=y)
    hf.create_dataset('longitude', data=x)
    hf.create_dataset('altitude', data=alts)
