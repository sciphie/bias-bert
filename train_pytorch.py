import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from rtpt import RTPT
import torch
import random, time, datetime
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

from transformers import AdamW, BertConfig, DistilBertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from train_functions import load_hf
import util as u

######################################
### ### ### PLEASE SPECIFY ### ### ### 
task_in = 'IMDB' 
#task_in = 'Twitter' 

model_id_in = "albertbase" # "bertbase", 'bertlarge', "distbase", "robertabase", "robertalarge", "albertbase", "albertlarge"
specs_all = ["original", "N_pro", "N_weat", "N_all", "mix_pro", "mix_weat", "mix_all"]

specs_in = specs_all # [specs_all[6]]
run_in = "ex_BS" #'ex_LR2'

i=0
# name_addition = "LR05" # None 
name_additions = ["LR1", "LR05", "LR5"]
learning_rates = [1e-5, 5e-6, 5e-5]
######################################


lr= learning_rates[i]
na= name_additions[i]

#####
assert(task_in in ['IMDB', 'Twitter']), 'task name is not valid'
assert(model_id_in in ["bertbase", 'bertlarge', "distbase", "distlarge", "robertabase", "robertalarge", "albertbase", "albertlarge"]), model_id_in + ' is not a valid model_id'
#print('called train.py {} {} {}'.format(task_in, model_id_in, specs_in))
#####    


def train(task=task_in, model_id=model_id_in, spec=specs_in[0],
          epochs = 20, max_len = 512,     # 
          lr_in = 2e-5, eps_in = 1e-8,    # for adam optimizer
          h_droo = 0.5, a_droo = 0.2,     # droppouts: hidden and attention 
          batch_s = 32, # 16, 8, 
          run=run_in, name_addition = None, name=None, rtpt_train=None):
    
    print('called train.py {} {} {}'.format(task, model_id, spec))
    t = datetime.datetime.now().strftime('%m_%d_%H%M')
    #u.check_path(run)

    if not name:
        #short_name = 'e{}do{}'.format(str(epochs),str(int(h_droo*10)))
        name = 'e{}do{}'.format(str(epochs),str(int(h_droo*10)))
        experiment_name = "{}_{}_{}_{}".format(task, model_id, spec, t)
        if name_addition:
            experiment_name = experiment_name + '_' + name_addition
        
    u.check_path('./res_models/{}/{}'.format(run, experiment_name))
    sys.stdout = open('./res_models/{}/{}/log_{}'.format(run, experiment_name, name), 'a')
        
    print('###################################')
    print('start new training')
    print('epochs = {}'.format(epochs, max_len, h_droo, a_droo, name, rtpt_train) )
    print('use GBU no. ' )# + str(i))

    #---------------------------------------------------------------------- 
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    #----------------------------------------------------------------------
    
    # +++++++++++++++++++++++++++++++++++++
    #           Model & Tokenizer
    # +++++++++++++++++++++++++++++++++++++    
    tokenizer, model = u.load_hf(model_id, h_droo)  # configuration=configuration)

    print("MODEL DOT CUDA")
    model.cuda()
    
    # +++++++++++++++++++++++++++++++++++++
    #           Data Preprocessing
    # +++++++++++++++++++++++++++++++++++++
    train_sentences, train_labels = u.import_data('train', task, spec)
    
    input_ids = []
    attention_masks = []

    for sent in train_sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_len,      # Pad & truncate all sentences.
                            pad_to_max_length = True,       # 
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence and its attention mask (simply differentiates padding from non-padding).
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(train_labels)

    # +++++++++
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # ++++++++++++
    # Create the DataLoaders for the training and validation sets.
    # Training samples are selected in random order; For validation the order doesn't matter. 
    train_dataloader = DataLoader(
                train_dataset,                          # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_s                    # Trains with this batch size.
            )

    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                # shuffle = True, # comment: You do not need to pass in a RandomSampler. You can specify "shuffle=True" to the DataLoader(), and it will create a RandomSampler for you.
                batch_size = batch_s # Evaluate with this batch size.
            )

    # ++++++++++++++
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    # ++++++++++++++
    optimizer = AdamW(model.parameters(),
                      lr = lr_in,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = eps_in # args.adam_epsilon  - default is 1e-8.
                    )

    # ++++++++++++++
    # Number of training epochs. The BERT authors recommend between 2 and 4. 
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    # epochs = eps_in

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # ++++++++++++++
    writer = SummaryWriter(u.check_path('res_models/{}/writer'.format(run)) + '/{}_{}'.format(experiment_name, name))


    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    
    if not rtpt_train:
        rtpt_train = RTPT(name_initials='SJ', experiment_name=name, max_iterations=epochs)
    rtpt_train.start()
    # For each epoch...
    for epoch_i in range(0, epochs):

        # +++++++++++++++++++++++++++++++++++++
        #              Training 
        # +++++++++++++++++++++++++++++++++++++
        print("")
        print('\n ======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. 
        # this does not start the training. 
        # `dropout` and `batchnorm` layers behave differently in both modes
        # (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = u.format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader by splittung it up 
            # and copy each tensor to the GPU using the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # function and pass down the arguments. The `forward` function is 
            # documented here: 
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            # Specifically, we'll get the loss (because we provided labels) and the
            # "logits"--the model outputs prior to activation.
            result = model(b_input_ids, 
                          # token_type_ids=None, 
                           attention_mask=b_input_mask, 
                           labels=b_labels,
                           return_dict=True)

            loss = result.loss
            logits = result.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = u.format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # +++++++++++++++++++++++++++++++++++++
        #              Validation 
        # +++++++++++++++++++++++++++++++++++++
        # After each training epoch, measure performance on validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        model.eval()

        # Tracking variables for tensorboard
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        accuracy_sk = 0
        recall_sk = 0
        precision_sk = 0
        f1_sk = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(b_input_ids, 
                               #token_type_ids=None, ###
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)

            # Get the loss and "logits" output by the model. The "logits" are the 
            # output values prior to applying an activation function like the softmax.
            loss = result.loss
            logits = result.logits

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches.
            total_eval_accuracy += u.flat_accuracy(logits, label_ids)

            ###### error
            # Classification metrics can't handle a mix of binary and continuous-multioutput targets
            logits_ = [0 if x>y else 1 for (x,y) in logits]
            accuracy_sk += accuracy_score(y_true=label_ids, y_pred=logits_) # 
            recall_sk += recall_score(y_true=label_ids, y_pred=logits_)
            precision_sk += precision_score(y_true=label_ids, y_pred=logits_)
            f1_sk += f1_score(y_true=label_ids, y_pred=logits_)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        curr_len = len(validation_dataloader)
        avg_val_loss = total_eval_loss / curr_len
        avg_accuracy_sk = accuracy_sk / curr_len
        avg_recall_sk = recall_sk / curr_len
        avg_precision_sk = precision_sk / curr_len
        avg_f1_sk = f1_sk / curr_len

        # Measure how long the validation run took.
        validation_time = u.format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        writer.add_scalar("train/Loss", avg_train_loss, epoch_i)
        writer.add_scalar("val/Loss", avg_val_loss, epoch_i)
        writer.add_scalar("val/Acc", avg_val_accuracy, epoch_i)
        writer.add_scalar("val/accuracy_sk", avg_accuracy_sk, epoch_i)
        writer.add_scalar("val/recall_sk", avg_recall_sk, epoch_i)
        writer.add_scalar("val/precision_sk", avg_precision_sk, epoch_i)
        writer.add_scalar("val/f1_sk", avg_f1_sk, epoch_i)

        training_stats.append({
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
        output_dir = './res_models/{}/{}/{}_epoch{}'.format(run, experiment_name , name, epoch_i)
        safe_model(output_dir, model, tokenizer)
        rtpt_train.step()
    print("Training complete!")
    writer.flush() # to make sure that all pending events have been written to disk
    writer.close()

    print("Total training took {:} (h:mm:ss)".format(u.format_time(time.time()-total_t0)))

    # Display floats with two decimal places.
    pd.set_option('precision', 2)

    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
    
    #####################################################
    ### loss plot: unnecessary when using tensorboard ###
    #####################################################
    
    # Display and save the table.
 #   print(df_stats)
 #   df_stats.to_pickle('res_models/{}/stats_{}_{}.pkl'.format(run, name, experiment_name))

 #   sns.set(style='darkgrid')

 #   sns.set(font_scale=1.5) # Increase the plot size and font size.
 #   plt.clf()
 #   plt.rcParams["figure.figsize"] = (12,6)
 #   plt.plot(df_stats['Training Loss'], 'b-o', label="Training Loss")
 #   plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation Loss")

 #   title = "Hidden Dropout= {}, Attention Drophout= {}".format(h_droo, a_droo)
 #   plt.title(title)
 #   plt.xlabel("Epoch")
 #   plt.ylabel("Loss")
 #   plt.legend()
 #   plt.ylim(top=2)
 #   plt.xticks([10, 20])
    
 #   u.check_path('./training_log/{}'.format(run))
 #   plt.savefig('./training_log/{}/loss_{}.png'.format(run, name))

    
def safe_model(output_dir, model, tokenizer ):
    print("Saving model to %s" % output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)# +'_tokenizer')
    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))


##############################################
### to start training directly from script ###
##############################################
#for (lr, na) in zip(learning_rates, name_additions):
#for spec in specs_in:
#    train(task=task_in, spec=spec, lr_in=lr, name_addition=na)

