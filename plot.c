#include <stdio.h>
#include <stdlib.h>

int main(void) {

    FILE *gp = popen("gnuplot -persist", "w"); // Open pipe to gnuplot
    if (gp == NULL) {
        printf("Error opening pipe to gnuplot.\n");
        exit(EXIT_FAILURE);
    }

    fprintf(gp,
        "set datafile commentschars '#'\n"
        "set title 'Test Accuracy per Epoch'\n"
        "set xlabel 'Epoch'\n"
        "set ylabel 'Accuracy'\n"
        "plot 'logs/agile_accuracy.dat' using 1:2 with lines title 'AGILE', \\\n"
        "     'logs/adam_accuracy.dat' using 1:2 with lines title 'Adam', \\\n"
        "     'logs/sgd_accuracy.dat' using 1:2 with lines title 'SGD'\n"
    );

    fflush(gp);

    pclose(gp); // Close pipe
    return 0;
}

