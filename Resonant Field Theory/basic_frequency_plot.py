import numpy as np
import matplotlib.pyplot as plt

def generate_frequency_analysis_plot(frequencies, amplitudes, output_path):
    \"\"\"Generates a basic frequency analysis plot with labeled peaks.\"\"\"
    # 1. Basic test, output frequencies to see where it goes in to
    print("Input frequencies:", frequencies)
    print("Input amplitudes:", amplitudes)

    # 2. Plot Frequency data (x) against amplitudes (y)
    plt.plot(frequencies, amplitudes, label='Frequency Spectrum')

    # 3. Make sure the frequencies line up with how is set in enhance_visualizations

    test_f_tetrahedron = 396 # Test variable

    #Add labels to make sure the code is output
    plt.annotate('T4_test', xy=(test_f_tetrahedron, 0.6), xytext=(test_f_tetrahedron+150, 0.7), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color = 'black')
    # Set label
    # Add grid, labels, title, and legend
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Basic Frequency Analysis Plot')
    plt.grid(True)
    plt.legend()

    # 4. Ensure the plots are correct, ensure that T4 is created at the right place
    print("Plot created")
    # Save the plot to a file
    plt.savefig(output_path)
    print("Plot data to: {0}".format(output_path))

    plt.close()

if __name__ == "__main__":
    # Test frequencies and amplitude
    test_frequencies = np.linspace(100, 1000, 50)
    test_amplitudes = np.random.rand(50)
    output_path = 'basic_frequency_plot.png'
    generate_frequency_analysis_plot(test_frequencies, test_amplitudes, output_path)
    print('Successfully plot and is runnable')

# Code to test.

