#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

const int ZERO = 0.00001;

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double gps_x, double gps_y, double theta, double std[]) {
    num_particles = 200;

    normal_distribution<double> dist_x(gps_x, std[0]);
    normal_distribution<double> dist_y(gps_y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Creation of the particles
    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = int(i);
        p.weight = 1.0;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);

        particles.push_back(p);
    }
    is_initialized = true;
}


// Move each particle according to bicycle motion model (taking noise into account)
void ParticleFilter::prediction(double dt, double std[], double velocity, double yaw_rate) {
    for (int i = 0; i < num_particles; ++i) {
        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;

        double new_theta, new_x, new_y;

        if (abs(yaw_rate) > ZERO) {
            new_theta = theta + yaw_rate * dt;
            new_x = x + velocity / yaw_rate * (sin(new_theta) - sin(theta));
            new_y = y + velocity / yaw_rate * (cos(theta) - cos(new_theta));
        } else {
            new_theta = theta;
            new_x = x + velocity * dt * cos(theta);
            new_y = y + velocity * dt * sin(theta);
        }

        normal_distribution<double> dist_x(new_x, std[0]);
        normal_distribution<double> dist_y(new_y, std[1]);
        normal_distribution<double> dist_theta(new_theta, std[2]);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    for (auto &obs : observations) {
        double min_dist = numeric_limits<double>::max();
        for (auto &pred : predicted) {
            double d = dist(obs.x, obs.y, pred.x, pred.y);
            if (d < min_dist) {
                obs.id   = pred.id;
                min_dist = d;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];

    for (int i = 0; i < num_particles; ++i) {
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        vector<LandmarkObs> seen_landmarks;

        // Selecting only those in sensor range
        for (const auto& map_landmark : map_landmarks.landmark_list) {
            int l_id   = map_landmark.id_i;
            double l_x = (double) map_landmark.x_f;
            double l_y = (double) map_landmark.y_f;

            double d = dist(p_x, p_y, l_x, l_y);
            if (d < sensor_range) {
                LandmarkObs l_pred;
                l_pred.id = l_id;
                l_pred.x = l_x;
                l_pred.y = l_y;
                seen_landmarks.push_back(l_pred);
            }
        }

        // Convert observations to map coordinate system
        vector<LandmarkObs> conn_landmarks;
        for (int j = 0; j < observations.size(); ++j) {
            LandmarkObs tmp;
            tmp.x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
            tmp.y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;

            conn_landmarks.push_back(tmp);
        }

        dataAssociation(seen_landmarks, conn_landmarks);
        double particle_prob = 1.0;

        // Calculating 
        double mu_x, mu_y;
        for (auto &obs : conn_landmarks) {
            for (auto &land: seen_landmarks)
                if (obs.id == land.id) {
                    mu_x = land.x;
                    mu_y = land.y;
                    break;
                }

            double bottom = 2 * M_PI * std_x * std_y;
            double prob = exp(-(pow(obs.x - mu_x, 2) / (2 * std_x * std_x) + pow(obs.y - mu_y, 2) / (2 * std_y * std_y)));

            particle_prob *= prob / bottom;
        }
        particles[i].weight = particle_prob;
    }

    double norm_factor = 0.0;
    for (auto &p : particles)
        norm_factor += p.weight;

    for (auto &p : particles)
        p.weight /= norm_factor;
}

void ParticleFilter::resample() {
    vector<double> ws;
    vector<Particle> resampled_particles;
    for (auto a : particles)
        ws.push_back(a.weight);
    discrete_distribution<int> d(ws.begin(), ws.end());

    for(int i = 0; i < num_particles; i++)
        resampled_particles.push_back(particles[d(gen)]);
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                        const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
