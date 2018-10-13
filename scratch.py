
def alpha_k(k, a_k, dim):
    if k == 2:
        ak = 1.0 - 3.0 / (4.0 * dim)
    else:
        ak1 = a_k[k-1]
        ak = ak1 + (1.0 - ak1) / 6.0
    a_k[k] = ak
    return ak


def cluster_eval(distortions, a_k, k, dim):
    if k == 1:
        return 1.0
    elif k == 2:
        a_k[1] = 0.0

    a_k[k] = alpha_k(k, a_k, dim)
    if a_k[k-1] != 0.0:
        return distortions[k] / (a_k[k-1] * distortions[k-1])
    else:
        return 1.0


def calculate_alpha_k(df, max_k):
    a_k = np.zeros(max_k)
    dims = len(df.columns)

    for k in range(2, max_k + 1):
        i = k - 1
        if k == 2:
            a_k[i] = 1.0 - 3.0 / (4.0 * dims)
        else:
            a_k[i] = a_k[i-1] + (1.0 - a_k[i-1]) / 6.0
    return a_k


def calculate_s_k(df, k):
    km = KMeans(n_clusters=k, random_state=42).fit(df)
    return km.inertia_ #-km.score(df) #


def calculate_s_k_array(df, max_k):
    s_k = np.zeros(max_k)

    for i in range(0, max_k):
        s_k[i] = calculate_s_k(df, i+1)
    return s_k


def calculate_f_k(df, max_k):
    distortions = DistortionCurve()
    a_k = dict()

    f_k = np.ones(max_k)

    dim = len(df.columns)

    for k in range(1, max_k + 1):
        i = k - 1
        #a_k[i] = alpha_k(k, a_k, dim)
        distortions[k] = calculate_s_k(df, k)
        f_k[i] = cluster_eval(distortions, a_k, k, dim)
    return f_k


def estimate_k(df, max_k):
    f_k = calculate_f_k(df, max_k)
    k = np.argmin(f_k)
    return k+1


def distance_to_line(x0, y0, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(dx * dx + dy * dy)


def estimate_elbow(df):
    max_dist = -1
    s_k_list = list()
    sk0 = 0

    for k in range(1, len(df) + 1):
        sk1 = calculate_s_k(df, k)
        s_k_list.append(sk1)
        if k > 2 and abs(sk0 - sk1) < 1e-3:
            break
        sk0 = sk1

    s_k = np.array(s_k_list)
    #print(s_k)

    #fig, ax1 = plt.subplots()
    #ax1.plot(range(1, len(s_k) + 1), s_k)
    #plt.show()

    x0 = 1
    y0 = s_k[0]

    x1 = len(s_k)
    y1 = 0

    print("(1; {0}) - ({1}; 0)".format(y0, x1))

    for k in range(1, len(s_k)):
        dist = distance_to_line(k, s_k[k-1], x0, y0, x1, y1)
        #print('({0}, {1}): {2}'.format(k, s_k[k-1], dist))
        if dist > max_dist:
            max_dist = dist
        else:
            return k - 1
    return -1
