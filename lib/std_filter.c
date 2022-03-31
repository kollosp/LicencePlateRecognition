#include <math.h>
#include <stdint.h>

int std_filter(
    double * buffer,
    intptr_t filter_size,
    double * return_value,
    void * user_data
) {
    double mu = 0;
    double std = 0;

    // Compute the empirical mean under the footprint
    for(int i=0; i<filter_size; i++)
        mu += buffer[i] / filter_size;

    /// Compute the empirical standard deviation under the footprint
    for(int i=0; i<filter_size; i++)
        std += (buffer[i] - mu)*(buffer[i] - mu) / (filter_size-1);

    *return_value = sqrt(std);
    return 1;
}