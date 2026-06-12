# CAffNet: Hard Constraint-Affine Neural Networks

## Authors

Yang Zhao<sup>1</sup>, Jungeun Lee<sup>2</sup>, Jeong hwan Jeon<sup>2</sup>, Sze Zheng Yong<sup>1</sup>

<sup>1</sup> Mechanical and Industrial Engineering, Northeastern University  
<sup>2</sup> Electrical Engineering, Ulsan National Institute of Science and Technology

## Abstract

We present a novel framework for embedding hard constraint satisfaction into neural network (NN) architectures, specifically feedforward neural networks and transformers, with input-dependent affine constraints of arbitrary cardinality. Traditional constraint enforcement approaches either rely on penalty-based soft constraints, which offer no guarantee of satisfaction, or on post-processing methods that enforce constraints after the NN is trained, which may lead to suboptimality. We introduce a trainable constraint-affine (CAffine) layer into NNs, yielding CAffNet, which goes beyond enforcing affine constraints via fixed orthogonal or parallel projections and enables joint optimization with network parameters. Moreover, we impose no restrictions on the constraint space dimensions and establish that our construction preserves the universal approximation properties of NNs, while providing provable guarantees on constraint adherence for all inputs. Experimental validation demonstrates robust performance across diverse domains requiring guaranteed constraint satisfaction.

## Paper

[Read the preprint](https://arxiv.org/abs/2605.24437)

## More Information

### CAffNet-Lite

CAffNet-Lite is a lightweight extension of CAffNet for safe-by-design neural network control.

- Introduces a lightweight constraint decomposition strategy that reduces computation while preserving hard constraint satisfaction and approximation performance.
- Jointly learns a neural network controller and neural-network-parameterized CBF parameters, avoiding an online QP safety filter at run time.
- [Read the preprint](https://arxiv.org/abs/2605.26534)

## Contact

- [Email](mailto:zhao.yang12@northeastern.edu?cc=jungeunlee@unist.ac.kr,jhjeon@unist.ac.kr,s.yong@northeastern.edu)
- [LinkedIn](https://www.linkedin.com/in/yang-zhao-b152991a2/)