from dataset import LargeMovieDataset




if __name__ == '__main__':
    # Create dataset
    dataset = LargeMovieDataset(path='../data/aclImdb_v1/aclImdb')
    print('Dataset size:', len(dataset))
    print('First sample:', dataset[0])
    print('Second sample:', dataset[1])
    print('Last sample:', dataset[-1])
    print('Second to last sample:', dataset[-2])