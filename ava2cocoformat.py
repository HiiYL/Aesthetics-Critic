from pandas import HDFStore
import nltk

store = HDFStore('data/labels.h5')

ava_train = store['labels_train']

comment_index = 1

lower_limit = 10
upper_limit = 20
annotations = []
images = []

iteration = 0
total_iterations = len(ava_train.comments)
for index, image_comment in ava_train.comments.iteritems():
    comments = image_comment.split(" [END] ")
    contain_valid_comment = False
    for comment in comments:
        tokens = nltk.tokenize.word_tokenize(comment.lower())
        if (len(tokens) >= lower_limit) and (len(tokens) < upper_limit) and ('challenge' not in tokens):
            item = {}
            item['caption']  = comment
            item['image_id'] = int(index)
            item['id'] = comment_index
            comment_index += 1
            annotations.append(item)
            contain_valid_comment = True

    if contain_valid_comment:    
        images.append({'id': int(index), 'file_name': "{}.jpg".format(index)}) 

    if iteration % 1000 == 0:
        print("[{}/{}] Processing Comments ... ".format(iteration, total_iterations))
    iteration += 1


train_json = {'images': images, 'annotations': annotations }

json.dump(train_json, open('data/aesthetics/captions_train.json', 'w'))
# for efficiency lets group annotations by image
itoa = {}
for a in annotations:
    imgid = a['image_id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a)