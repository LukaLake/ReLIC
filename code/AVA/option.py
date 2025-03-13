import argparse

def init():
    parser = argparse.ArgumentParser(description="PyTorch")
    parser.add_argument('--path_to_ava_txt', type=str, default="C:/Users/Administrator/Documents/GitHub/ReLIC/data/AVA/AVA.txt",
                        help='directory to csv_folder')

    parser.add_argument('--path_to_images', type=str, default=r'D:\Datasets\AVA_dataset\image',
                        help='directory to images for training')
    
    parser.add_argument('--path_to_test_images', type=str, default=r'C:\Users\Administrator\Documents\GitHub\ReLIC\Images',
                        help='directory to images for prediction')
    
    parser.add_argument('--image_name', type=str, default='69.jpg',
                        help='image name')

    parser.add_argument('--path_to_save_csv', type=str,default=r"C:\Users\Administrator\Documents\GitHub\ReLIC/data/AVA/",
                        help='directory to csv_folder')

    parser.add_argument('--experiment_dir_name', type=str, default='.',
                        help='directory to project')

    parser.add_argument('--path_to_model_weight', type=str, default=r'C:\Users\Administrator\Documents\GitHub\ReLIC/code/AVA/pretrain_model/relic2_model.pth',
                        help='directory to pretrain model')
    
    parser.add_argument('--path_to_teacher_model_weight', type=str, default=r'C:\Users\Administrator\Documents\GitHub\ReLIC/code/AVA/pretrain_model/relic2_model.pth',
                        help='directory to pretrained teacher model')
    
    parser.add_argument('--path_to_student_model_weight', type=str, default=r'C:\Users\Administrator\Documents\GitHub\ReLIC\code\AVA\trained_models\student_model_best.pth',
                        help='directory to trained student model')

    parser.add_argument('--init_lr', type=int, default=0.00003, help='learning_rate'
                        )
    parser.add_argument('--num_epoch', type=int, default=30, help='epoch num for train'
                        )
    parser.add_argument('--batch_size', type=int,default=32,help='how many pictures to process one time'
                        )
    parser.add_argument('--num_workers', type=int, default=8, help ='num_workers',
                        )
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')


    args = parser.parse_args()
    return args