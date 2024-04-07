/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	//输入参数：当前线程在其协同网格中的索引;球谐系数的层数;球谐系数size(0)个数；三维坐标；相机的光心；全部的球谐系数
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];//三维坐标
	glm::vec3 dir = pos - campos;//相对于相机光心的坐标
	dir = dir / glm::length(dir);//相对于相机光心的方向

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	//上面一坨就是球谐函数的累加
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	//clamped就是存储了颜色是不是为正
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
// 计算二维协方差矩阵，利用雅可比矩阵
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	//三维点世界坐标，x焦距，y焦距，x视野tan，y视野tan，协方差矩阵，外参矩阵
	//论文式29和式31
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.

	float3 t = transformPoint4x3(mean, viewmatrix);//相机坐标下的三维坐标

	const float limx = 1.3f * tan_fovx;//这个似乎会比fov实际的视角大一点
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;//将范围限制在正负limxy内
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);//计算雅可比矩阵

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };//最后要用的是一个对称的2d矩阵
	//总体计算参考https://blog.csdn.net/qq_50791664/article/details/135932903
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
// 通过存储的rotation和scale计算三维的协方差矩阵
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
//scale，尺度参数（1），rot，协方差矩阵地址
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);//3x3的矩阵
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);//四元数转旋转矩阵

	glm::mat3 M = S * R;//旋转后再缩放

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;//应该转置在后 R S S^t R^t，这里转置在前不知道为什么
	//3d高斯和椭球是等价的，可以利用旋转后放缩来表示
	//后续优化也是用椭球的旋转和放缩来优化的

	// Covariance is symmetric, only store upper right
	// 协方差是对称的，只存储右上角
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>//模版参数，图像的层数，RGB就是三层
//每个点一个线程
__global__ void preprocessCUDA(int P, int D, int M,//多少个点，球谐系数的层数，球谐系数size(0)个数
	const float* orig_points,//三维点的三维坐标
	const glm::vec3* scales,//scale参数，暂时不知道是干嘛的，应该和三维高斯有关，维度为N
	const float scale_modifier,//应该是尺度参数，默认为1
	const glm::vec4* rotations,//rotation参数，暂时不知道是干嘛的，应该和三维高斯有关，维度为N
	const float* opacities,//三维点的不透明度
	const float* shs,//全部的球谐系数
	bool* clamped,
	const float* cov3D_precomp,//预先计算好的协方差矩阵，单纯渲染的时候为空，后续计算的
	const float* colors_precomp,//在python中为colors_precomp，输入不是RGB则必须，单纯渲染的时候是空
	const float* viewmatrix,//相机外参矩阵；
	const float* projmatrix,//投影矩阵，内参矩阵和外参矩阵一通计算得到
	const glm::vec3* cam_pos,//相机的光心，用位姿反算出来的
	const int W, int H,//图像高宽
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,//xy的焦距
	int* radii,//和三维点size相同，都是0
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,//网格尺寸
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();//似乎可以直接返回当前线程在其协同网格中的索引
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))//判断是否在视野内
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);//计算得到相机裁剪空间中的点p_proj
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };//归一化

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;//预先计算好直接用idx定位
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
	//输入参数：三维点世界坐标，x焦距，y焦距，x视野tan，y视野tan，协方差矩阵，外参矩阵
	//三维投影到二维的椭圆可以用一个2x2的对称矩阵表示（三个有效值）
	//(x,y),(y,z)

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);//协方差矩阵的行列式
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };//逆矩阵

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	//这里计算了上面那个2d协方差矩阵的特征值，其中较大的那个代表了椭圆的长轴
	//这里三倍就是正负三个标准差，覆盖绝大部分高斯
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	//ndc空间投影到像素空间
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;
		//如果左右 or 上下边界相等（椭圆等于0不存在）

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		//用球谐函数计算了颜色值
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		//输入参数：当前线程在其协同网格中的索引;球谐系数的层数;球谐系数size(0)个数；三维坐标；相机的光心；全部的球谐系数
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;//深度
	radii[idx] = my_radius;//三维高斯投影到二维后长轴的值
	points_xy_image[idx] = point_image;//像素坐标
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);//覆盖网格的面积
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,//存储了每个瓦片上高斯idx的开头结尾
	const uint32_t* __restrict__ point_list,//高斯的索引
	int W, int H,
	const float2* __restrict__ points_xy_image,//像素坐标
	const float* __restrict__ features,//颜色
	const float4* __restrict__ conic_opacity,//conic和透明度
	float* __restrict__ final_T,//img属性
	uint32_t* __restrict__ n_contrib,//img属性
	const float* __restrict__ bg_color,//背景颜色
	float* __restrict__ out_color)//输出颜色
{
	// Identify current tile and associated min/max pixel range.
	// 识别当前瓦片和相关的最小/最大像素范围
	auto block = cg::this_thread_block();//索引
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;//水平最多几个砖块
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };//当前线程块的起始像素坐标
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };//当前线程块的结束像素坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };//当前线程的的像素坐标
	uint32_t pix_id = W * pix.y + pix.x;//当前线程的一维像素索引
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];//获取当前瓦片的range，高斯的idx的头尾
	//下面一切的操作都是在当前瓦片上进行的
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);//需要开几个BLOCKSIZE
	int toDo = range.y - range.x;//要做几个splat

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];//高斯索引
	__shared__ float2 collected_xy[BLOCK_SIZE];//像素坐标
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];//协方差矩阵的逆矩阵
	//开辟共享内存

	// Initialize helper variables
	//初始化变量
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)//每个循环splat BLOCKSIZE 个高斯
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		//线程同步。同时统计了已经做完（同步）的点
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		//从全局到共享的每个高斯数据的集体提取
		int progress = i * BLOCK_SIZE + block.thread_rank();//计算当前线程的全局位置
		if (range.x + progress < range.y)//是否超过需要处理的范围
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;//高斯的索引
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			//都存储到共享内存中
		}
		block.sync();//线程同步

		// Iterate over current batch
		// 在当前批次上迭代
		//下面就是正式的渲染过程
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)//一般一次循环一个BLOCKSIZE，最后一个循环比较小，考虑toDo
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };//当前线程和高斯中心的距离
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;//高斯分布的概率的计算公式（差个exp在下面）
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));//这个高斯分布分布到这个像素点上的不透明度，α
			if (alpha < 1.0f / 255.0f)//太透明
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;//沿着光线一直算，如果基本已经不透光了就不继续算了
			}

			// Eq. (3) from 3D Gaussian splatting paper.、
			// α-blending
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;//每次循环套一个1-α

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];//最后和背景融合
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,//有多少个瓦片(网格尺寸),瓦片尺寸(线程块尺寸)
	const uint2* ranges,//存储了每个瓦片上高斯idx的开头结尾
	const uint32_t* point_list,////高斯的索引
	int W, int H,//宽高
	const float2* means2D,//像素坐标
	const float* colors,//颜色
	const float4* conic_opacity,//conic和透明度
	float* final_T,//img属性
	uint32_t* n_contrib,//img属性
	const float* bg_color,//背景颜色
	float* out_color)//输出颜色
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}
//预处理
void FORWARD::preprocess(int P, int D, int M,//多少个点，球谐系数的层数，球谐系数size(0)个数
	const float* means3D,//三维点的三维坐标
	const glm::vec3* scales,//scale参数，暂时不知道是干嘛的，应该和三维高斯有关，维度为N
	const float scale_modifier,//应该是尺度参数，默认为1
	const glm::vec4* rotations,//rotation参数，暂时不知道是干嘛的，应该和三维高斯有关，维度为N
	const float* opacities,//三维点的不透明度
	const float* shs,//全部的球谐系数
	bool* clamped,
	const float* cov3D_precomp,//预先计算好的协方差矩阵，单纯渲染的时候为空，后续计算的
	const float* colors_precomp,//在python中为colors_precomp，输入不是RGB则必须，单纯渲染的时候是空
	const float* viewmatrix,//相机外参矩阵；
	const float* projmatrix,//投影矩阵，内参矩阵和外参矩阵一通计算得到
	const glm::vec3* cam_pos,//相机的光心，用位姿反算出来的
	const int W, int H,//图像高宽
	const float focal_x, float focal_y,////xy的焦距
	const float tan_fovx, float tan_fovy,
	int* radii,//和三维点size相同，都是0
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,//网格尺寸
	uint32_t* tiles_touched,
	bool prefiltered)//默认false
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (//这里就开始调用cuda核函数了，<NUM_CHANNELS>是模版参数，为3
																//(P + 255) / 256个线程块，每个线程块256个线程
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}