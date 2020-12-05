#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

std::vector<double> getNextLineAndSplitIntoTokens(std::istream &str) {
  std::vector<double> result;
  std::string line;
  std::getline(str, line);

  std::stringstream lineStream(line);
  std::string cell;

  while (std::getline(lineStream, cell, ',')) {
    result.push_back(std::stod(cell));
  }
  // This checks for a trailing comma with no data after it.
  if (!lineStream && cell.empty()) {
    // If there was a trailing comma then add an empty element.
    // result.push_back("");
  }
  return result;
}

class System {
public:
  System(const std::string &input_file) {
    std::ifstream f;
    f.open(input_file);

    if (!f) {
      std::cerr << "Unable to open file " << input_file << std::endl;
      exit(EXIT_FAILURE);
    }

    // read data from file and construct matrix
    std::string line;
    std::getline(f, line);

    m_system_size = std::stod(line);

    m_variable_value_t = new double[m_system_size];
    m_variable_value_prev_t = new double[m_system_size];

    m_value_matrix = new double *[m_system_size];
    for (int i = 0; i < m_system_size; i++) {
      m_value_matrix[i] = new double[m_system_size];
    }

    std::vector<double> ret = getNextLineAndSplitIntoTokens(f);

    int lineCounter = 0;
    while (ret.size() != 0) {

      for (int i = 0; i < m_system_size; i++) {
        if (lineCounter == m_system_size) {
          m_variable_value_prev_t[i] = ret[i];
        } else {
          m_value_matrix[lineCounter][i] = ret[i];
        }
      }
      ret = getNextLineAndSplitIntoTokens(f);

      lineCounter++;
    }
  };

  int size() const { return m_system_size; }

  void solve(int n_steps) {
    // perform n steps of simulation
    for (int t = 0; t < n_steps; ++t) {
      double min = m_variable_value_prev_t[0];
      double max = m_variable_value_prev_t[0];
      double range;

      // identify range of system values
#pragma omp parallel for reduction(min:min) reduction(max:max) 
      for (int i = 0; i < m_system_size; ++i) {
        if (m_variable_value_prev_t[i] < min) {
          min = m_variable_value_prev_t[i];
        }
        if (m_variable_value_prev_t[i] > max) {
          max = m_variable_value_prev_t[i];
        }
      }
      range = max - min;

      // update variable values
#pragma omp parallel default(none) shared(t, min, range)
      {
	// divide work among threads
        int iam = omp_get_thread_num();
        int nt = omp_get_num_threads();
        // size of partition
        int ipoints = m_system_size / nt;
        // starting array index
        int istart = iam * ipoints;
        // last thread may do more
        if (iam == nt - 1) {
          ipoints = m_system_size - istart;
        }
        partial_update(istart, ipoints, min, range);
      }

      // store values for next iteration
 #pragma omp parallel for
      for (int i = 0; i < m_system_size; ++i) {
        m_variable_value_prev_t[i] = m_variable_value_t[i];
      }
    }
  }

  bool check_correctness(const double *validation_matrix) const {
    bool system_is_valid = true;
    double range = 1000000.0;
    for (int i = 0; i < m_system_size; i++) {
      // round 6 digit after comma
      double valid = round(validation_matrix[i] * range) / range;
      double result = round(m_variable_value_prev_t[i] * range) / range;

      if (valid != result) {
        system_is_valid = false;
        fprintf(stderr, "%d -- %10.10e != %10.10e (%10.10e)\n", i, valid,
                result, m_variable_value_prev_t[i]);
      }
    }

    if (system_is_valid)
      std::cerr << "System is valid" << std::endl;
    else {
      std::cerr << "System is not valid" << std::endl;
    }
    return system_is_valid;
  }

  void partial_update(int start, int npoints, double min, double range) {
    for (int i = 0; i < npoints; ++i) {
      double updated_value = 0;
      const int offset = start+i;
      for (int j = 0; j < m_system_size; ++j) {
	updated_value += m_variable_value_prev_t[j] * m_value_matrix[offset][j];
      }
      // normalize
      m_variable_value_t[offset] = (updated_value-min)/range;
    }
  }

private:
  double *m_variable_value_t;
  double *m_variable_value_prev_t;
  double **m_value_matrix;
  int m_system_size;
};

int main(int argc, char **argv) {
  // check command line arguments
  if (argc < 4) {
    std::cerr << "USAGE: " << argv[0]
              << " input_file nb_steps validation_file\n"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // number of timesteps
  const int n_steps = std::atoi(argv[2]);

  // file containing input data
  const std::string input_filename = argv[1];
  System system(input_filename);

  // file containing validation data
  const std::string validation_filename = argv[3];
  std::ifstream validation_file;
  validation_file.open(validation_filename);

  if (!validation_file) {
    std::cerr << "Unable to open file " << validation_filename << std::endl;
    exit(EXIT_FAILURE);
  }

  double *validation_matrix = new double[system.size()];
  std::vector<double> ret = getNextLineAndSplitIntoTokens(validation_file);
  for (int i = 0; i < system.size(); i++) {
    validation_matrix[i] = ret[i];
  }

  // solve system
  auto t1 = std::chrono::high_resolution_clock::now();
  system.solve(n_steps);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << std::getenv("OMP_NUM_THREADS") << "," << system.size() << "," << n_steps << "," << duration
            << std::endl;

  // check system solution
  bool correct = system.check_correctness(validation_matrix);
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
