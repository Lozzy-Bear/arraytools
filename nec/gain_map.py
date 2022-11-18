import matplotlib.pyplot as plt
import numpy as np
import pretty_plots
import pymap3d as pm


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
    tx_data, _ = read_nec('ibtx4l.out')
    rx_data, _ = read_nec('ibrx.out')

    y = np.arange(40, 64, 0.1)
    x = np.arange(-125, -90, 0.1)
    lons, lats = np.meshgrid(x, y)
    alt = 130.0e3
    m = np.zeros(lons.shape)
    print('mshape', m.shape)
    tx_lat = 50.893
    tx_lon = -109.403
    rx_lat = 52.243
    rx_lon = -106.450

    # az1, el1, r1 = pm.geodetic2aer(60.0, -108.403, alt,
    #                                tx_lat, tx_lon, 0.0, ell=pm.Ellipsoid("wgs84"), deg=True)
    # print(90.0 - el1)
    # exit()

    for i in range(lons.shape[0]):
        for j in range(lats.shape[1]):
            az1, el1, r1 = pm.geodetic2aer(lats[i, j], lons[i, j], alt,
                                           tx_lat, tx_lon, 0.0, ell=pm.Ellipsoid("wgs84"), deg=True)
            az2, el2, r2 = pm.geodetic2aer(lats[i, j], lons[i, j], alt,
                                           rx_lat, rx_lon, 0.0, ell=pm.Ellipsoid("wgs84"), deg=True)
            az1 = np.round(((90.0 + 16.0 - az1) % 360.0))
            az2 = np.round(((90.0 + 7.0 - az2) % 360.0))
            el1 = np.round(90.0 - el1)
            el2 = np.round(90.0 - el2)
            pw1 = np.argwhere((tx_data[0, :] == el1) & (tx_data[1, :] == az1))
            pw2 = np.argwhere((rx_data[0, :] == el2) & (rx_data[1, :] == az2))
            pw = tx_data[4, pw1] + rx_data[4, pw2]
            # print(i, j, pw, pw.size, az1, el1, az2, el2)
            # if (el1 < 0.0) or (el2 < 0.0):
            #     pw = np.nan
            if pw.size == 0:
                pw = np.nan
            else:
                pw = pw.flatten()[0]
            m[i, j] = pw

    plt.figure()
    plt.imshow(m, origin='lower', cmap='inferno', vmin=-20)
    plt.colorbar()
    plt.xticks(np.linspace(0, 350, 35), labels=np.arange(-125, -90, 1))
    plt.yticks(np.linspace(0, 240, 24), labels=np.arange(40, 64, 1))
    plt.grid()

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

    plt.show()

