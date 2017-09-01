import numpy as np
import tensorflow as tf

from config import cfg
from model import FCR_aGAN
from util import DataProcess, scene_model_id_pair, onehot, scene_model_id_pair_test
from sklearn.metrics import average_precision_score
import copy

def evaluate(batch_size, checknum, mode):

    n_vox = cfg.CONST.N_VOX
    dim = cfg.NET.DIM
    vox_shape = [n_vox[0],n_vox[1],n_vox[2],dim[4]]
    dim_z = cfg.NET.DIM_Z
    start_vox_size = cfg.NET.START_VOX 
    kernel = cfg.NET.KERNEL
    stride = cfg.NET.STRIDE
    freq = cfg.CHECK_FREQ
    refine_ch = cfg.NET.REFINE_CH
    refine_kernel = cfg.NET.REFINE_KERNEL

    save_path = cfg.DIR.EVAL_PATH
    chckpt_path = cfg.DIR.CHECK_PT_PATH + str(checknum) + '-' + str(checknum * freq)

    fcr_agan_model = FCR_aGAN(
                batch_size=batch_size,
                vox_shape=vox_shape,
                dim_z=dim_z,
                dim=dim,
                start_vox_size=start_vox_size,
                kernel=kernel,
                stride=stride,
                refine_ch=refine_ch,
                refine_kernel = refine_kernel,
                )


    Z_tf, z_enc_tf, vox_tf, vox_gen_tf, vox_gen_decode_tf, vox_refine_dec_tf, vox_refine_gen_tf,\
     recons_loss_tf, code_encode_loss_tf, gen_loss_tf, discrim_loss_tf, recons_loss_refine_tfs, gen_loss_refine_tf, discrim_loss_refine_tf,\
      cost_enc_tf, cost_code_tf, cost_gen_tf, cost_discrim_tf, cost_gen_ref_tf, cost_discrim_ref_tf, summary_tf = fcr_agan_model.build_model()
    Z_tf_sample, vox_tf_sample = fcr_agan_model.samples_generator(visual_size=batch_size)
    sample_vox_tf, sample_refine_vox_tf = fcr_agan_model.refine_generator(visual_size=batch_size)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    # Restore variables from disk.
    saver.restore(sess, chckpt_path)

    print("...Weights restored.")

    if mode == 'recons':
        #reconstruction and generation from normal distribution evaluation
        #generator from random distribution
        for i in np.arange(batch_size):
        	Z_np_sample=np.random.normal(size=(1,start_vox_size[0],start_vox_size[1],start_vox_size[2],dim_z)).astype(np.float32)
        	if i == 0:
        	    Z_var_np_sample = Z_np_sample
        	else:
        	    Z_var_np_sample = np.concatenate((Z_var_np_sample, Z_np_sample), axis=0)
        np.save(save_path + '/sample_z.npy', Z_var_np_sample)

        generated_voxs_fromrand=sess.run(
            vox_tf_sample,
            feed_dict={Z_tf_sample:Z_var_np_sample})
        vox_models_cat = np.argmax(generated_voxs_fromrand, axis=4)
        np.save(save_path + '/generate.npy', vox_models_cat)

        refined_voxs_fromrand = sess.run(
            sample_refine_vox_tf,
            feed_dict={sample_vox_tf:generated_voxs_fromrand})
        vox_models_cat = np.argmax(refined_voxs_fromrand, axis=4)
        np.save(save_path + '/generate_refine.npy', vox_models_cat)

        #evaluation for reconstruction
        voxel_test, num = scene_model_id_pair_test(dataset_portion=cfg.TRAIN.DATASET_PORTION)
        num = voxel_test.shape[0]
        print("test voxels loaded")
        for i in np.arange(int(num/batch_size)):
            batch_voxel_test = voxel_test[i*batch_size:i*batch_size+batch_size]

            batch_generated_voxs, batch_enc_Z = sess.run(
                [vox_gen_decode_tf, z_enc_tf],
                feed_dict={vox_tf:batch_voxel_test})
            batch_refined_vox = sess.run(
                sample_refine_vox_tf,
                feed_dict={sample_vox_tf:batch_generated_voxs})

            if i == 0:
                generated_voxs = batch_generated_voxs
                refined_voxs = batch_refined_vox
                enc_Z = batch_enc_Z
            else:
                generated_voxs = np.concatenate((generated_voxs, batch_generated_voxs), axis=0)
                refined_voxs = np.concatenate((refined_voxs, batch_refined_vox), axis=0)
                enc_Z = np.concatenate((enc_Z, batch_enc_Z), axis=0)

        print("forwarded")

        #real
        vox_models_cat = voxel_test
        np.save(save_path + '/real.npy', vox_models_cat)

        #decoded
        vox_models_cat = np.argmax(generated_voxs, axis=4)
        np.save(save_path + '/recons.npy', vox_models_cat)
        vox_models_cat = np.argmax(refined_voxs, axis=4)
        np.save(save_path + '/recons_refine.npy', vox_models_cat)
        np.save(save_path + '/decode_z.npy', enc_Z)

        print("voxels saved")


        #numerical evalutation
        on_real = onehot(voxel_test,vox_shape[3])
        on_recons = onehot(np.argmax(generated_voxs, axis=4),vox_shape[3])

        #calc_IoU
        IoU_class = np.zeros([vox_shape[3]+1])
        for class_n in np.arange(vox_shape[3]):
            on_recons_ = on_recons[:,:,:,:,class_n]
            on_real_ = on_real[:,:,:,:,class_n]
            mother = np.sum(np.add(on_recons_, on_real_),(1,2,3))
            child = np.sum(np.multiply(on_recons_, on_real_),(1,2,3))
            count = 0
            IoU_element = 0
            for i in np.arange(num):
                if mother[i] != 0:
                    IoU_element += child[i]/mother[i]
                    count += 1
            IoU_calc = np.round(IoU_element/count,3)
            IoU_class[class_n] = IoU_calc
            print 'IoU class ' + str(class_n) + '=' + str(IoU_calc)

        on_recons_ = on_recons[:,:,:,:,1:vox_shape[3]]
        on_real_ = on_real[:,:,:,:,1:vox_shape[3]]
        mother = np.sum(np.add(on_recons_, on_real_),(1,2,3,4))
        child = np.sum(np.multiply(on_recons_, on_real_),(1,2,3,4))
        count = 0
        IoU_element = 0
        for i in np.arange(num):
            if mother[i] != 0:
                IoU_element += child[i]/mother[i]
                count += 1
        IoU_calc = np.round(IoU_element/count,3)
        IoU_class[vox_shape[3]] = IoU_calc
        print 'IoU all =' + str(IoU_calc)
        np.savetxt(save_path + '/IoU.csv', IoU_class, delimiter=",")


        #calc_AP
        AP_class = np.zeros([vox_shape[3]+1])
        for class_n in np.arange(vox_shape[3]):
            on_recons_ = generated_voxs[:,:,:,:,class_n]
            on_real_ = on_real[:,:,:,:,class_n]

            AP = 0.
            for i in np.arange(num):
                y_true = np.reshape(on_real_[i],[-1])
                y_scores = np.reshape(on_recons_[i],[-1])
                if np.sum(y_true) > 0.:
                    AP += average_precision_score(y_true, y_scores)
            AP = np.round(AP/num,3)
            AP_class[class_n] = AP
            print 'AP class ' + str(class_n) + '=' + str(AP)

        on_recons_ = generated_voxs[:,:,:,:,1:vox_shape[3]]
        on_real_ = on_real[:,:,:,:,1:vox_shape[3]]
        AP = 0.
        for i in np.arange(num):
            y_true = np.reshape(on_real_[i],[-1])
            y_scores = np.reshape(on_recons_[i],[-1])
            if np.sum(y_true) > 0.:
                AP += average_precision_score(y_true, y_scores)

        AP = np.round(AP/num,3)
        AP_class[vox_shape[3]] = AP
        print 'AP all =' + str(AP)
        np.savetxt(save_path + '/AP.csv', AP_class, delimiter=",")

        #Refine
        #calc_IoU
        on_recons = onehot(np.argmax(refined_voxs, axis=4),vox_shape[3])

        IoU_class = np.zeros([vox_shape[3]+1])
        for class_n in np.arange(vox_shape[3]):
            on_recons_ = on_recons[:,:,:,:,class_n]
            on_real_ = on_real[:,:,:,:,class_n]
            mother = np.sum(np.add(on_recons_, on_real_),(1,2,3))
            child = np.sum(np.multiply(on_recons_, on_real_),(1,2,3))
            count = 0
            IoU_element = 0
            for i in np.arange(num):
                if mother[i] != 0:
                    IoU_element += child[i]/mother[i]
                    count += 1
            IoU_calc = np.round(IoU_element/count,3)
            IoU_class[class_n] = IoU_calc
            print 'IoU class ' + str(class_n) + '=' + str(IoU_calc)

        on_recons_ = on_recons[:,:,:,:,1:vox_shape[3]]
        on_real_ = on_real[:,:,:,:,1:vox_shape[3]]
        mother = np.sum(np.add(on_recons_, on_real_),(1,2,3,4))
        child = np.sum(np.multiply(on_recons_, on_real_),(1,2,3,4))
        count = 0
        IoU_element = 0
        for i in np.arange(num):
            if mother[i] != 0:
                IoU_element += child[i]/mother[i]
                count += 1
        IoU_calc = np.round(IoU_element/count,3)
        IoU_class[vox_shape[3]] = IoU_calc
        print 'IoU all =' + str(IoU_calc)
        np.savetxt(save_path + '/IoU_refine.csv', IoU_class, delimiter=",")

        #calc_AP
        AP_class = np.zeros([vox_shape[3]+1])
        for class_n in np.arange(vox_shape[3]):
            on_recons_ = refined_voxs[:,:,:,:,class_n]
            on_real_ = on_real[:,:,:,:,class_n]

            AP = 0.
            for i in np.arange(num):
                y_true = np.reshape(on_real_[i],[-1])
                y_scores = np.reshape(on_recons_[i],[-1])
                if np.sum(y_true) > 0.:
                    AP += average_precision_score(y_true, y_scores)
            AP = np.round(AP/num,3)
            AP_class[class_n] = AP
            print 'AP class ' + str(class_n) + '=' + str(AP)

        on_recons_ = refined_voxs[:,:,:,:,1:vox_shape[3]]
        on_real_ = on_real[:,:,:,:,1:vox_shape[3]]
        AP = 0.
        for i in np.arange(num):
            y_true = np.reshape(on_real_[i],[-1])
            y_scores = np.reshape(on_recons_[i],[-1])
            if np.sum(y_true) > 0.:
                AP += average_precision_score(y_true, y_scores)

        AP = np.round(AP/num,3)
        AP_class[vox_shape[3]] = AP
        print 'AP all =' + str(AP)
        np.savetxt(save_path + '/AP_refine.csv', AP_class, delimiter=",")



    #interpolation evaluation
    if mode == 'interpolate':
        interpolate_num = 30
        #interpolatioin latent vectores
        decode_z = np.load(save_path + '/decode_z.npy')
        decode_z = decode_z[:batch_size]
        for l in np.arange(batch_size):
            for r in np.arange(batch_size):
                if l != r:
                    print l,r
                    base_num_left = l
                    base_num_right = r
                    left = np.reshape(decode_z[base_num_left], [1,start_vox_size[0],start_vox_size[1],start_vox_size[2],dim_z])
                    right = np.reshape(decode_z[base_num_right], [1,start_vox_size[0],start_vox_size[1],start_vox_size[2],dim_z])

                    duration = (right - left)/(interpolate_num-1)
                    if base_num_left == 0:
                        Z_np_sample = decode_z[1:]
                    elif base_num_left == batch_size-1:
                        Z_np_sample = decode_z[:batch_size-1]
                    else:
                        Z_np_sample_before = np.reshape(decode_z[:base_num_left], [base_num_left,start_vox_size[0],start_vox_size[1],start_vox_size[2],dim_z])
                        Z_np_sample_after = np.reshape(decode_z[base_num_left+1:], [batch_size-base_num_left-1,start_vox_size[0],start_vox_size[1],start_vox_size[2],dim_z])
                        Z_np_sample = np.concatenate([Z_np_sample_before, Z_np_sample_after], axis=0)
                    for i in np.arange(interpolate_num):
                        if i == 0:
                            Z = copy.copy(left)
                            interpolate_z = copy.copy(Z)
                        else:
                            Z=Z + duration
                            interpolate_z = np.concatenate([interpolate_z, Z], axis = 0)
                        Z_var_np_sample = np.concatenate([Z, Z_np_sample], axis=0)
                        generated_voxs_fromrand=sess.run(
                            vox_tf_sample,
                            feed_dict={Z_tf_sample:Z_var_np_sample})
                        refined_voxs_fromrand=sess.run(
                            sample_refine_vox_tf,
                            feed_dict={sample_vox_tf:generated_voxs_fromrand})
                        interpolate_vox = np.reshape(refined_voxs_fromrand[0], [1,vox_shape[0],vox_shape[1],vox_shape[2],vox_shape[3]])
                        if i == 0:
                            generated_voxs = interpolate_vox
                        else:
                            generated_voxs = np.concatenate([generated_voxs, interpolate_vox], axis=0)
                    
                    np.save(save_path + '/interpolation_z' + str(l) + '-' + str(r) + '.npy', interpolate_z)
                    
                    vox_models_cat = np.argmax(generated_voxs, axis=4)
                    np.save(save_path + '/interpolation' + str(l) + '-' + str(r) + '.npy', vox_models_cat)
        print("voxels saved")

    #add noise evaluation
    if mode == 'noise':
        decode_z = np.load(save_path + '/decode_z.npy')
        decode_z = decode_z[:batch_size]
        noise_num = 10
        for base_num in np.arange(batch_size):
            print base_num
            base = np.reshape(decode_z[base_num], [1,start_vox_size[0],start_vox_size[1],start_vox_size[2],dim_z])
            eps = np.random.normal(size=(noise_num-1,dim_z)).astype(np.float32)
            
            if base_num == 0:
                Z_np_sample = decode_z[1:]
            elif base_num == batch_size-1:
                Z_np_sample = decode_z[:batch_size-1]
            else:
                Z_np_sample_before = np.reshape(decode_z[:base_num], [base_num,start_vox_size[0],start_vox_size[1],start_vox_size[2],dim_z])
                Z_np_sample_after = np.reshape(decode_z[base_num+1:], [batch_size-base_num-1,start_vox_size[0],start_vox_size[1],start_vox_size[2],dim_z])
                Z_np_sample = np.concatenate([Z_np_sample_before, Z_np_sample_after], axis=0)

       
            for c in np.arange(start_vox_size[0]):
                for l in np.arange(start_vox_size[1]):
                    for d in np.arange(start_vox_size[2]):

                        for i in np.arange(noise_num):
                            if i == 0:
                                Z = copy.copy(base)
                                noise_z = copy.copy(Z)
                            else:
                                Z = copy.copy(base)
                                Z[0,c,l,d,:] += eps[i-1]
                                noise_z = np.concatenate([noise_z, Z], axis = 0)
                            Z_var_np_sample = np.concatenate([Z, Z_np_sample], axis=0)
                            generated_voxs_fromrand=sess.run(
                                vox_tf_sample,
                                feed_dict={Z_tf_sample:Z_var_np_sample})
                            refined_voxs_fromrand=sess.run(
                                sample_refine_vox_tf,
                                feed_dict={sample_vox_tf:generated_voxs_fromrand})
                            noise_vox = np.reshape(refined_voxs_fromrand[0], [1,vox_shape[0],vox_shape[1],vox_shape[2],vox_shape[3]])
                            if i == 0:
                                generated_voxs = noise_vox
                            else:
                                generated_voxs = np.concatenate([generated_voxs, noise_vox], axis=0)
                        
                        np.save(save_path + '/noise_z' + str(base_num) + '_' + str(c) + str(l) + str(d) + '.npy', noise_z)

                        
                        vox_models_cat = np.argmax(generated_voxs, axis=4)
                        np.save(save_path + '/noise' + str(base_num) +'_' + str(c) + str(l) + str(d) + '.npy', vox_models_cat)


        print("voxels saved")



