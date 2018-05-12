function imdb = cub(opts, net)

CUBdir = '/research/object_detection/data/CUB200-2011';

% train/test split: first 100 train, last 100 test
[id, fn] = textread([CUBdir '/images.txt'], '%d %s');
labels = cellfun(@(x) sscanf(x(1:3), '%d'), fn);
set(labels <= 100) = 1;
set(labels >  100) = 3;

% class names
[~, cls_name] = textread([CUBdir '/classes.txt'], '%d %s');

% load, reshape
imgSize = opts.imageSize;
data = cellfun(@(x) imread([CUBdir '/images/' x]), fn, 'uniform', false);
data = cellfun(@(x) imresize(x, [imgSize imgSize]), data, 'uniform', false);
data = cat(4, data{:});
size(data)

imdb.images.data = single(data) ;
imdb.images.labels = single(labels) ;
imdb.images.set = uint8(set) ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = cls_name;
end
