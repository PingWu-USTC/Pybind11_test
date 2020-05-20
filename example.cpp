#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<vector>
#include<complex> 
#include<math.h>
const double pi = acos(-1.0); 
using std::vector;
namespace py = pybind11;
/*尽情复制pybind11给出的解决方案吧*/
py::array_t<double> Fourier_transform(py::array_t<double> input1, py::array_t<double>input2)
{
	py::buffer_info buf1 = input1.request();
	py::buffer_info buf2 = input2.request();
	if (buf1.ndim != 2 || buf2.ndim != 3)
	{
		throw std::runtime_error("the dimension of input is not correct!,please try again!");
	}
	auto source = input1.mutable_unchecked<2>();
	auto U_field = input2.mutable_unchecked<3>();
	ssize_t len_x = U_field.shape(0);
	ssize_t len_y = U_field.shape(1);
	ssize_t len_z = U_field.shape(2);
	std::vector<std::vector<std::vector<double>>> cosXPart(len_x, std::vector<std::vector<double>>(len_x, std::vector<double>(len_y, 0)));
	std::vector<std::vector<std::vector<double>>> sinXPart(len_x, std::vector<std::vector<double>>(len_x, std::vector<double>(len_y, 0)));
	std::vector<std::vector<std::vector<double>>> cosYPart(len_x, std::vector<std::vector<double>>(len_x, std::vector<double>(len_y, 0)));
	std::vector<std::vector<std::vector<double>>> sinYPart(len_x, std::vector<std::vector<double>>(len_x, std::vector<double>(len_y, 0)));
	std::vector<std::vector<std::vector<double>>> rMinusU(len_x, std::vector<std::vector<double>>(len_y, std::vector<double>(len_z, 0)));

	/*这样定义数组的维数感觉有点奇葩*/
	for (ssize_t i = 0; i < len_x; i++)
	{
		for (ssize_t j = 0; j < len_y; j++)
		{
			for (ssize_t k = 0; k < len_z; k++)
			{
				rMinusU[i][j][0] = U_field(i, j, 0) + i;
				rMinusU[i][j][1] = U_field(i, j, 1) + j;
			}
		}
	}

	/*下午来写fourier_transform部分*/
	for (ssize_t k = 0; k < len_x; k++)
	{
		double qx = k * 2 * pi / len_x;
		for (ssize_t i = 0; i < len_x; i++)
		{
			for (ssize_t j = 0; j < len_y; j++)
			{
				cosXPart[k][i][j] = cos(qx * rMinusU[i][j][0]);
				sinXPart[k][i][j] = sin(qx * rMinusU[i][j][0]);
			}
		}
	}
	for (ssize_t k = 0; k < len_y; k++)
	{
		double qy = k * 2 * pi / len_x;
		for (ssize_t i = 0; i < len_x; i++)
		{
			for (ssize_t j = 0; j < len_y; j++)
			{
				cosYPart[k][i][j] = cos(qy * rMinusU[i][j][1]);
				sinYPart[k][i][j] = sin(qy * rMinusU[i][j][1]);
			}
		}
	}
	for (ssize_t i = 0; i < len_x; i++)
	{
		for (ssize_t j = 0; j < len_y; j++)
		{
			if ((rMinusU[i][j][0] < 0) || (rMinusU[i][j][0] > len_x - 1) || (rMinusU[i][j][1] < 0) || (rMinusU[i][j][1] > len_y - 1))
			{
				source(i, j) = 0;
			}
		}
	}
	std::vector<std::vector<std::vector<double>>> fftz(len_x, std::vector<std::vector<double>>(len_y, std::vector<double>(2, 0)));

	for (ssize_t i = 0; i < len_x; i++)
	{
		for (ssize_t j = 0; j < len_y; j++)
		{
			for (ssize_t p = 0; p < len_x; p++)
			{
				for (ssize_t q = 0; q < len_y; q++)
				{
					fftz[i][j][0] += source(p, q) * (cosXPart[i][p][q] * cosYPart[j][p][q] - sinXPart[i][p][q] * sinYPart[j][p][q]);
					fftz[i][j][1] += -source(p, q) * (sinXPart[i][p][q] * cosYPart[j][p][q] + cosXPart[i][p][q] * sinYPart[j][p][q]);
				}

			}
		}
	}
	std::vector<std::vector<std::vector<double>>> fftz_center(len_x, std::vector<std::vector<double>>(len_y, std::vector<double>(2, 0)));
	for (ssize_t i = 0; i < len_x; i++)
	{
		for (ssize_t j = 0; j < len_y; j++)
		{
			ssize_t i_new = (i + (int)(len_x / 2)) % len_x;
			ssize_t j_new = (j + (int)(len_y / 2)) % len_y;
			fftz_center[i_new][j_new][0] = fftz[i][j][0];
			fftz_center[i_new][j_new][1] = fftz[i][j][1];
		}
	}
	auto result = py::array_t<double>(buf2.size);
	result.resize({ buf2.shape[0], buf2.shape[1], buf2.shape[2] });
	py::buffer_info buf_result = result.request();
	auto r3 = result.mutable_unchecked<3>();
	for (ssize_t i = 0; i < len_x; i++)
	{
		for (ssize_t j = 0; j < len_y; j++)
		{
			for (ssize_t k = 0; k < len_z; k++)
			{
				r3(i,j,k) = fftz[i][j][k];
			}
		}
	}
	return result;
}
py::array_t<double> getDeviationGaussDefault(double L, py::array_t<double> input1,py::array_t<double>input2)
{
	py::buffer_info buf1 = input1.request();
	py::buffer_info buf2 = input2.request();

	auto cosmsintopo = input1.mutable_unchecked<3>();
	auto gauss = input2.mutable_unchecked<2>();
	ssize_t N = cosmsintopo.shape(0);
	ssize_t nx = cosmsintopo.shape(0);
	ssize_t ny = cosmsintopo.shape(1);
	auto result = py::array_t<double>(buf1.size);
	result.resize({ buf1.shape[0], buf1.shape[1], buf1.shape[2] });
	py::buffer_info buf_result = result.request();
	
	auto expu = result.mutable_unchecked<3>();

	for (ssize_t i = 0; i < nx; i++)
	{
		for (ssize_t j = 0; j < ny; j++)
		{
			double gsum = 0;
			expu(i, j, 0) = 0;
			expu(i, j, 1) = 1;
			ssize_t x = i;
			ssize_t y = j;
			ssize_t xpmin = round(fmax(0, x - 3 * L - 1));
			ssize_t xpmax = round(fmin(N, x + 3 * L + 1));
			ssize_t ypmin = round(fmax(0, y - 3 * L - 1));
			ssize_t ypmax = round(fmin(N, y + 3 * L + 1));
			for (ssize_t xprime = xpmin; xprime < xpmax; xprime++)
			{
				for (ssize_t yprime = ypmin; yprime < ypmax; yprime++)
				{
					ssize_t x_1 = (int)abs(x - xprime);
					ssize_t y_1 = (int)abs(y - yprime);
					gsum += gauss(x_1, y_1);
					double cmplex_x = cosmsintopo(xprime, yprime, 0) * gauss(x_1, y_1);
					double cmplex_y = cosmsintopo(xprime, yprime, 1) * gauss(x_1, y_1);
					expu(i, j, 0) += cmplex_x;
					expu(i, j, 1) += cmplex_y;
				}
			}
			expu(i, j, 0) /= gsum;
			expu(i, j, 1) /= gsum;
		}
	}
	return result;
}
PYBIND11_MODULE(example, m)
{
	m.doc() = "pybind11 fftz_test";
	m.def("Fourier_transform", &Fourier_transform);
	m.def("getDeviationGaussDefault", &getDeviationGaussDefault);
}