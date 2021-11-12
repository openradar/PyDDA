import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import copy
import tensorflow as tf
from ..cost_functions._cost_functions_auglag import *

def auglag_function(winds, parameters, mult, mu, resto):
    winds_t = tf.constant(winds, dtype=tf.float64)
    al = 0
    al_grad = tf.zeros(len(winds), dtype=tf.float64)

    if not resto:
        Jvel, Jvel_grad = radial_velocity_function(winds_t, parameters)
        al += Jvel
        al_grad += Jvel_grad

    if resto:
        if parameters.Cm > 0:
            mult.mass_cont = tf.zeros(mult.mass_cont.shape, dtype=tf.float64)
        if parameters.Cv > 0:
            mult.vert_vort = tf.zeros(mult.vert_vort.shape, dtype=tf.float64)
        mu = 1.0

    if parameters.Cm > 0:
        Jmass, Jmass_grad = al_mass_cont_function(winds_t, parameters, mult.mass_cont, mu)
        al += Jmass
        al_grad += Jmass_grad

    if parameters.Cv > 0:
        Jvort, Jvort_grad = al_vert_vort_function(winds_t, parameters, mult.vert_vort, mu)
        al += Jvort
        al_grad += Jvort_grad

    return al.numpy(), al_grad.numpy().astype('float64')


class Filter:
    def __init__(self, winds, cv0, g0, Jvel0, beta = 0.9, gamma = 0.9):
        self.cvs = cv0
        self.gs = g0
        self.Jvels = Jvel0
        self.cv_min = self.cvs
        self.g_min = self.gs
        self.beta = beta
        self.gamma = gamma
        self.sols = winds

    def add_to_filter(self, winds, cv, g, Jvel):
        self.cvs = np.append(self.cvs,cv)
        self.gs = np.append(self.gs, g)
        self.sols = np.vstack((self.sols,winds))
        self.Jvels = np.append(self.Jvels,Jvel)
        if g < self.g_min:
            self.g_min = g
        if self.cv_min == 0 or cv < self.cv_min:
            self.cv_min = cv

    def check_acceptable(self, cv, g):
        cond1 = (cv <= self.beta*self.cvs)
        cond2 = (g <= (self.gs - self.gamma*cv))
        acceptable = np.logical_or(cond1,cond2)

        return acceptable.all()

class StopOptimizingException(Exception):
    pass

class Callback:
    def __init__(self, al, g, cv0, Jvel, AL_Filter, obj_func, obj_func_zero, parameters,theta = 30.0):
        self.obj_func = obj_func
        self.obj_func_zero = obj_func_zero
        self.AL_Filter = AL_Filter
        self.parameters = parameters
        self.cv0 = cv0
        self.gnew = g
        self.alnew = al
        self.g_mu = -1
        self.g0 = g
        self.Jvel0 = Jvel
        self.theta = theta
        self.target = al - theta*g

    def __call__(self,xk):
        alnew, al_grad = self.obj_func(xk, self.parameters)
        al_grad = tf.reshape(al_grad,
                             (3, self.parameters.grid_shape[0], self.parameters.grid_shape[1], self.parameters.grid_shape[2]))
        edge_boolean = False * np.ones(self.parameters.grid_shape)
        edge_boolean[2, -1, :, :] = True
        edge_boolean[2, 0, :, :] = True
        al_grad = tf.where(edge_boolean, 0., al_grad)
        gnew = tf.norm(tf.reshape(al_grad, (np.prod(al_grad.shape), )))
        infnorm = tf.norm(tf.reshape(al_grad, (np.prod(al_grad.shape), )), np.Inf)
        self.g_mu = gnew
        self.alnew = alnew
        alnewzero, al_grad_zero = self.obj_func_zero(xk,self.parameters)
        al_grad_zero = tf.reshape(al_grad, (3, self.parameters.grid_shape[0], self.parameters.grid_shape[1], self.parameters.grid_shape[2]))
        al_grad = tf.where(edge_boolean, 0., al_grad)
        self.gnew = np.linalg.norm(tf.reshape(al_grad_zero, (np.prod(al_grad.shape), )))
        winds = tf.reshape(xk,
                           (3, self.parameters.grid_shape[0], self.parameters.grid_shape[1], self.parameters.grid_shape[2]))
        
        cv2 = 0.0
        if self.parameters.Cm > 0:
            div = calculate_mass_continuity(winds[0], winds[1], winds[2],
                                            self.parameters.z,self.parameters.dx,self.parameters.dy,self.parameters.dz)
            cv2 += tf.norm(tf.reshape(div, np.prod(dif.shape,)))**2
        if self.parameters.Cv > 0:
            vort = calculate_vertical_vorticity(winds[0], winds[1], winds[2],
                                                self.parameters.dx, self.parameters.dy, self.parameters.dz,
                                                self.parameters.Ut,self.parameters.Vt)
            cv2 += tf.norm(tf.reshape(vort, np.prod(dif.shape,)))**2
        cv = np.sqrt(cv2)

        if (alnew <= self.target):
            self.winds = xk
            self.alnew = alnew
            self.cv = cv
            raise StopOptimizingException()
            return True
        else:
            return False

class RestoCallback:
    def __init__(self, AL_Filter, obj_func_zero, parameters):
        self.AL_Filter = AL_Filter
        self.obj_func_zero = obj_func_zero
        self.parameters = parameters

    def __call__(self,xk):
        alnew, al_grad = self.obj_func_zero(xk,self.parameters)
        g = tf.norm(al_grad)
        winds = tf.reshape(xk,
                           (3, self.parameters.grid_shape[0], self.parameters.grid_shape[1], self.parameters.grid_shape[2]))
        # COMPUTE TOTAL CONSTRAINT VIOLATION
        cv2 = 0
        if self.parameters.Cm > 0:
            div = calculate_mass_continuity(winds[0], winds[1], winds[2],
                                            self.parameters.z,
                                            self.parameters.dx, self.parameters.dy, self.parameters.dz)
            cv2 += tf.norm(tf.reshape(div, np.prod(div.shape,)))**2
        if self.parameters.Cv > 0:
            vort = calculate_vertical_vorticity(winds[0], winds[1], winds[2],
                                                self.parameters.dx,self.parameters.dy,self.parameters.dz,
                                                self.parameters.Ut,self.parameters.Vt)
            cv2 += tf.norm(tf.reshape(vort, np.prod(vort.shape,)))**2
        cv = tf.sqrt(cv2)
        self.cv = cv
        self.g = g
        self.al = alnew
        if self.AL_Filter.check_acceptable(cv, g):
            self.winds = xk
            raise StopOptimizingException()
            return True
        else:
            return False

class Multipliers(object):
    pass

def auglag(winds, parameters, bounds):

    mu = parameters.Cm
    
    # stopping criteria:
    cvtol = parameters.cvtol # maximum constraint violation must be less than this number (abs value of largest divergence must be less than this many 1/s units)
    gtol = parameters.gtol # Augmented Lagrangian norm must be less than this number
    Jveltol = parameters.Jveltol # acceptable terminating value of Jvel

    n = len(winds)

    # generate a random initial point
    winds = np.reshape(winds, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))

    # ensure initial point satisfies impermeability condition:
    winds[2,-1,:,:] = 0
    winds[2,0,:,:] = 0
    
    # initialize a (very coarse) guess of Lagrange multipliers
    mults = Multipliers()
    cv02 = 0.0
    if parameters.Cm > 0:
        div = calculate_mass_continuity(winds[0], winds[1], winds[2],
                                        parameters.z, parameters.dx, parameters.dy, parameters.dz).numpy()
        mults.mass_cont = -mu * div.flatten()
        cv02 += np.linalg.norm(div.flatten()) ** 2
    if parameters.Cv > 0:
        vort = calculate_vertical_vorticity(winds[0], winds[1], winds[2],
                                            parameters.dx, parameters.dy, parameters.dz,
                                            parameters.Ut, parameters.Vt).numpy()
        mults.vert_vort = -mu * vort.flatten()
        cv02 += np.linalg.norm(vort.flatten()) ** 2
    winds = winds.flatten()
    
    cv = np.sqrt(cv02)
    print("Initial constraint violation: ", "{:.6f}".format(cv))
    
    # initialize filter
    resto = False
    obj_func = lambda winds, parameters: auglag_function(winds, parameters, mults, 0.0, resto)
    al, al_grad = obj_func(winds, parameters)
    al_grad = np.reshape(al_grad, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
    al_grad[2,-1,:,:] = 0
    al_grad[2,0,:,:] = 0
    g = np.linalg.norm(al_grad)
    print("Initial Lagrangian norm: ", "{:.6f}".format(g))
    Jvel, Jvelgrad = radial_velocity_function(winds, parameters)
    AL_Filter = Filter(winds, cv, g, Jvel)
    
    multk = copy.deepcopy(mults)

    iter_count = 1
    funcalls = 0
    while True:
        while True:
            # run L-BFGS-B with current fixed values of mult and mu 
            obj_func = lambda winds, parameters: auglag_function(winds, parameters, mults, mu, resto)
            al_mu, al_grad = obj_func(winds.flatten(), parameters)
            al_grad = np.reshape(al_grad, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
            al_grad[2,-1,:,:] = 0
            al_grad[2,0,:,:] = 0
            g_mu = np.linalg.norm(al_grad)
            obj_func_zero = lambda winds, parameters: auglag_function(winds, parameters, mults, 0.0, resto)
            try:
                if iter_count > 1:
                    cb = Callback(al_mu, g_mu, cv, Jvel, AL_Filter, obj_func, obj_func_zero, parameters)
                    winds = fmin_l_bfgs_b(obj_func, winds, args=(parameters,), pgtol = gtol, maxiter=100,
                                          bounds=bounds, approx_grad=False, disp=1, iprint=-1)
                    # IF WE CHOOSE NOT TO ACTUALLY USE THE CALLBACK ABOVE:
                    alnew, al_grad = obj_func(winds[0],parameters)
                    al_grad = np.reshape(al_grad, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
                    al_grad[2,-1,:,:] = 0
                    al_grad[2,0,:,:] = 0
                    cb.g_mu = np.linalg.norm(al_grad.flatten())
                    alnew, al_grad = obj_func_zero(winds[0],parameters)
                    al_grad = np.reshape(al_grad, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
                    al_grad[2,-1,:,:] = 0
                    al_grad[2,0,:,:] = 0
                    cb.gnew = np.linalg.norm(al_grad.flatten())
                else:
                    cb = Callback(al_mu, g_mu, cv, Jvel, AL_Filter, obj_func, obj_func_zero, parameters)
                    winds = fmin_l_bfgs_b(obj_func, winds, args=(parameters,), pgtol = gtol,
                                          bounds=bounds, approx_grad=False, disp=1,iprint=-1)
                    alnew, al_grad = obj_func(winds[0],parameters)
                    al_grad = np.reshape(al_grad, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
                    al_grad[2,-1,:,:] = 0
                    al_grad[2,0,:,:] = 0
                    cb.g_mu = np.linalg.norm(al_grad.flatten())
                    alnew, al_grad = obj_func_zero(winds[0],parameters)
                    al_grad = np.reshape(al_grad, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
                    al_grad[2,-1,:,:] = 0
                    al_grad[2,0,:,:] = 0
                    cb.gnew = np.linalg.norm(al_grad.flatten())
                funcalls += winds[2]['funcalls']
                winds = np.reshape(winds[0], (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
            except StopOptimizingException:
                winds = cb.winds
                winds = np.reshape(winds, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
 
            g = cb.gnew
            if cb.g_mu >= 0:
                g_mu = cb.g_mu
                
            # compute constraint violation and Lagrangian stationary measure:
            cv2 = 0.0
            if parameters.Cm > 0:
                div = calculate_mass_continuity(winds[0],winds[1],winds[2],
                                                parameters.z,parameters.dx,parameters.dy,parameters.dz).numpy()
                div = div.flatten()
                cv2 += np.linalg.norm(div) ** 2
            if parameters.Cv > 0:
                vort = calculate_vertical_vorticity(winds[0], winds[1], winds[2], parameters.dx, parameters.dy, 
                                                    parameters.dz, parameters.Ut, parameters.Vt).numpy()
                vort = vort.flatten()
                cv2 += np.linalg.norm(vort) ** 2
            cv = np.sqrt(cv2)

            # check if restoration is necessary:
            if AL_Filter.beta * np.maximum(AL_Filter.g_min / AL_Filter.gamma,AL_Filter.beta*AL_Filter.cv_min) <= cv or \
                    (g_mu <= gtol and cv >= AL_Filter.beta * AL_Filter.cv_min):
                # increase penalty
                mu = 10.0 * mu
                # run L-BFGS-B to minimize constraint violation
                print("Restoration phase, mu = :", mu)
                obj_func = lambda winds, parameters: auglag_function(winds, parameters, mults, mu, False)
                obj_func_resto = lambda winds, parameters: auglag_function(winds, parameters, mults, mu, True)
                resto_cb = RestoCallback(AL_Filter, obj_func, parameters)
                try:
                    winds = fmin_l_bfgs_b(obj_func_resto, winds, args=(parameters,), pgtol=0, callback=resto_cb,
                                          bounds=bounds, approx_grad=False, disp=1, iprint=-1)
                    funcalls += winds[2]['funcalls']
                    winds = np.reshape(winds[0],
                                       (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
                except StopOptimizingException:
                    winds = resto_cb.winds
                    winds = np.reshape(winds,
                                       (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
                try:
                    cv = resto_cb.cv
                    g = resto_cb.g
                except:
                    print("Can't make progress in restoration, ending prematurely")
                    return winds, mults, AL_Filter, funcalls
            else:
                # update multipliers
                if parameters.Cm > 0:
                    mults.mass_cont = multk.mass_cont - mu*div
                if parameters.Cv > 0:
                    mults.vert_vort = multk.vert_vort - mu*vort
        
            # print some progress stats
            Jvel, Jvelgrad = radial_velocity_function(winds, parameters)
            print('Iter: ',iter_count)
            iter_count += 1
            print('Jvel: ', Jvel)
            print('Constraint violation: ', cv)
            viols = np.array([])
            if parameters.Cm > 0:
                maxviol = np.linalg.norm(div.flatten(), np.Inf)
                print("Maximum mass continuity violation: ", maxviol)
                viols = np.append(viols, maxviol)
            if parameters.Cv > 0:
                maxviol = np.linalg.norm(vort.flatten(), np.Inf)
                print("Maximum vertical vorticity violation: ", maxviol)
                viols = np.append(viols, maxviol)
            maxviol = np.amax(viols)

            # check if acceptable to filter
            obj_func_zero = lambda winds, parameters: auglag_function(winds, parameters, mults, 0.0, resto)
            alnew, al_grad = obj_func_zero(winds.flatten(), parameters)
            al_grad = np.reshape(al_grad,
                                 (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
            al_grad[2, -1, :, :] = 0
            al_grad[2, 0, :, :] = 0
            g = np.linalg.norm(al_grad.flatten())
            print("Lagrangian norm: ", "{:.6f}".format(g))
            if AL_Filter.check_acceptable(cv,g) or (maxviol <= cvtol and g <= gtol) \
                    or (maxviol <= cvtol and Jvel <= Jveltol):
                break

        # check stopping criteria
        if (maxviol <= cvtol and g <= gtol) or (maxviol <= cvtol and Jvel <= Jveltol):
            print('AugLag converged to specified tolerance')
            AL_Filter.add_to_filter(winds.flatten(),cv, g, Jvel)
            break

        # add newest point to filter
        AL_Filter.add_to_filter(winds.flatten(), cv, g, Jvel)
        print("Added most recent point to filter")
        multk = copy.deepcopy(mults)

    return winds, mults, AL_Filter, funcalls






