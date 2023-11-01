import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import shuffle

def rotate(raw, center = np.array([0.5, 0.5])):
    
    raw = np.array(raw)
    new = np.zeros_like(raw)
    angle = np.random.randint(low = -180, high = 180)
    
    new[:,0] = (raw[:,0] - center[0]) * np.cos(angle) - (raw[:,1] - center[1]) * np.sin(angle) + center[0]
    new[:,1] = (raw[:,0] - center[0]) * np.sin(angle) + (raw[:,1] - center[1]) * np.cos(angle) + center[1]
    
    minmum = np.minimum(np.min(new), 0)
    new -= minmum
    maximum = np.maximum(np.max(new), 1)
    new /= maximum
    #print(angle, minmum, maximum)
    return new


class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

# Load TSP train data
class GoogleTSPReader(object):
    """Iterator that reads TSP dataset files and yields mini-batches.
    
    Format expected as in Vinyals et al., 2015: https://arxiv.org/abs/1506.03134, http://goo.gl/NDcOIG
    Pointer Network
    """

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath, shuffled = True, augmentation = False, aug_prob = 0.9):
        """
        Args:
            num_nodes: Number of nodes in TSP tours
            num_neighbors: Number of neighbors to consider for each node in graph
            batch_size: Batch size
            filepath: Path to dataset file (.txt file)
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath
        self.augmentation = augmentation
        self.aug_prob = aug_prob
        if shuffled:
            self.filedata = shuffle(open(filepath, "r").readlines())  # Always shuffle upon reading data
        else:
            self.filedata = open(filepath, "r").readlines()
        self.max_iter = (len(self.filedata) // batch_size)

    def __iter__(self):
        for batch in range(self.max_iter):
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            yield self.process_batch(self.filedata[start_idx:end_idx])

    def process_batch(self, lines):
        """Helper function to convert raw lines into a mini-batch as a DotDict.
        """
        batch_edges = []  # 存放batch_size个TSP instance的邻接矩阵
        batch_edges_values = []  # 存放batch_size个TSP instance的带权重的邻接矩阵
        batch_edges_target = []  # Binary classification targets (0/1)
        batch_nodes = []  # 其实没有用到
        batch_nodes_target = []  # Multi-class classification targets (`num_nodes` classes)  # 存放batch个路由节点的顺序
        batch_nodes_coord = []  # 存放batch_size个节点的坐标
        batch_tour_nodes = []  #  存放batch个tsp instance的最优路径的节点
        batch_tour_len = []  # 存放batch个tsp instance的路由长度

        for line_num, line in enumerate(lines):
            line = line.split(" ")  # Split into list
            
            # Compute signal on nodes
            nodes = np.ones(self.num_nodes)  # All 1s for TSP...  [1,1,...1] len=20
            
            # Convert node coordinates to required format
            # nodes_coord = [[x1,y1],[x2,y2],...,[x20,y20]]
            nodes_coord = []
            for idx in range(0, 2 * self.num_nodes, 2):
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])
            if self.augmentation:  # 是否进行数据增强
                if np.random.uniform() > self.aug_prob:
                    #print(f'aug = {a}')
                    nodes_coord = rotate(raw = nodes_coord)
                #else:
                    #print(f'non-aug = {a}')
                
            # Compute distance matrix
            W_val = squareform(pdist(nodes_coord, metric='euclidean'))
            
            # Compute adjacency matrix
            if self.num_neighbors == -1:
                W = np.ones((self.num_nodes, self.num_nodes))  # Graph is fully connected
            else:
                W = np.zeros((self.num_nodes, self.num_nodes))
                # Determine k-nearest neighbors for each node
                knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]
                # Make connections 
                for idx in range(self.num_nodes):
                    W[idx][knns[idx]] = 1
            np.fill_diagonal(W, 2)  # Special token for self-connections 填充对角线元素为2
            
            # Convert tour nodes to required format
            # Don't add final connection for tour/cycle
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
            
            # Compute node and edge representation of tour + tour_len
            tour_len = 0
            nodes_target = np.zeros(self.num_nodes) # [0,0,0,...0] len=20
            edges_target = np.zeros((self.num_nodes, self.num_nodes))  # 20x20
            for idx in range(len(tour_nodes) - 1):
                i = tour_nodes[idx]
                j = tour_nodes[idx + 1]
                nodes_target[i] = idx  # node targets: ordering of nodes in tour
                edges_target[i][j] = 1
                edges_target[j][i] = 1
                tour_len += W_val[i][j]
            
            # Add final connection of tour in edge target
            nodes_target[j] = len(tour_nodes) - 1
            edges_target[j][tour_nodes[0]] = 1  # 将tsp instance最后访问的节点与最开始访问的节点相连
            edges_target[tour_nodes[0]][j] = 1  # 将tsp instance最后访问的节点与最开始访问的节点相连
            tour_len += W_val[j][tour_nodes[0]]
            
            # Concatenate the data
            batch_edges.append(W)  # batch个tsp instance的邻接矩阵
            batch_edges_values.append(W_val)  # batch个tsp instance节点之间的距离
            batch_edges_target.append(edges_target)  # batch个tsp instance的最优路径
            batch_nodes.append(nodes)
            batch_nodes_target.append(nodes_target)  # batch个tsp instance的节点的顺序
            batch_nodes_coord.append(nodes_coord)   # batch个tsp instance的节点的坐标
            batch_tour_nodes.append(tour_nodes)  #batch个tap instance的最优路径的节点
            batch_tour_len.append(tour_len)  # batch个tsp instance的最优路径值
        
        # From list to tensors as a DotDict
        batch = DotDict()
        batch.edges = np.stack(batch_edges, axis=0)
        batch.edges_values = np.stack(batch_edges_values, axis=0)
        batch.edges_target = np.stack(batch_edges_target, axis=0)
        batch.nodes = np.stack(batch_nodes, axis=0)
        batch.nodes_target = np.stack(batch_nodes_target, axis=0)
        batch.nodes_coord = np.stack(batch_nodes_coord, axis=0)
        batch.tour_nodes = np.stack(batch_tour_nodes, axis=0)
        batch.tour_len = np.stack(batch_tour_len, axis=0)
        return batch
