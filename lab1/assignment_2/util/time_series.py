def next_in_series(x, t, series):
    return x + (0.2 * get_x_from_series(series, t - 25)) / (1 + get_x_from_series(series, t - 25) ** 10) - 0.1 * x


def get_x_from_series(series, t):
    if t < 0:
        return 0
    else:
        return series[t]


def generate_time_series(starting_x=1.5):
    # generate time series
    time_series = [starting_x]
    for i in range(1505):
        time_series.append(next_in_series(time_series[i], i, series))
    print(time_series)
    print(time_series[281::5])

    generated_input = [[], [], [], [], []]
    generated_output = []
    # generate inputs and outputs
    for i in range(301, 1501):
        generated_input[0].append(time_series[i - 20])
        generated_input[1].append(time_series[i - 15])
        generated_input[2].append(time_series[i - 10])
        generated_input[3].append(time_series[i - 5])
        generated_input[4].append(time_series[i])
        generated_output.append(time_series[i + 5])
    for row in generated_input:
        print(row)
    print(generated_output)

    return generated_input, generated_output


def generate_time_series_transposed(starting_x=1.5):
    # generate time series
    time_series = [starting_x]
    for i in range(1505):
        time_series.append(next_in_series(time_series[i], i, time_series))
    print(time_series)
    print(time_series[281::5])

    generated_input = []
    generated_output = []
    # generate inputs and outputs
    for i in range(301, 1501):
        generated_input.append([time_series[i - 20],
                                time_series[i - 15],
                                time_series[i - 10],
                                time_series[i - 5],
                                time_series[i]])
        generated_output.append(time_series[i + 5])
    for row in generated_input:
        print(row)
    print(generated_output)

    return generated_input, generated_output
