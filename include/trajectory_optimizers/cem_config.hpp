// namespace trajectory_optimizers{
//     class cem_config{
//         public:
//             cem_config(int state_dimension,  int number_of_samples, 
//                 int number_of_time, int number_of_elite, int max_it, 
//                 double converge_r, 
//                 double mu_u, double std_u, 
//                 double mu_t, double std_t, 
//                 double t_max, double dt, double *loss_weights_ptr) : state_dimension(state_dimension), 
//                 ns(number_of_samples),
//                 nt(number_of_time),
//                 ne(number_of_elite),
//                 max_it(max_it),
//                 converge_r(converge_r),
//                 mu_u(mu_u), std_u(std_u),
//                 mu_t(mu_t), std_t(std_t),
//                 t_max(t_max), dt(dt) {
//                     loss_weights = new double[state_dimension];
//                     for(int i = 0; i < state_dimension; i++){
//                         loss_weights[i] = loss_weights_ptr[i];
//                     }

//                 };
//             ~cem_config(){
//                 delete loss_weights;
//             };
//             double* loss_weights;
//             int state_dimension, ns, nt, ne, max_it;                
//             double converge_r, mu_u, std_u, mu_t, std_t, t_max, dt;
            
//     };
// }

// trajectory_optimizers::cem_config get_default_arobot_cem_config(){
//     double loss_weights_ptr[4] = {1., 1., 0.3, 0.3};
//     return trajectory_optimizers::cem_config(
//         4, 1024, 5, 32, 20, 
//         0.1, 
//         0, 4, 
//         0.05, 0.2, 
//         0.5, 2e-2, loss_weights_ptr);
// }

