#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define T_ANGLE_THRE 10
#define T_SIZE_THRE 5
#define T_LUMI_THRE 50

void BrightAdjust(IplImage* src, IplImage* dst,
	double dContrast, double dBright)//1��-120
{
	int nVal;

	unsigned char* SrcData = (unsigned char*)src->imageData;//��������ͼƬsrc��ͼƬ�洢�ռ��е�����imagedata��ֵ��srcdata
	unsigned char* DstData = (unsigned char*)dst->imageData;//ͬ��
	int step = src->widthStep / sizeof(unsigned char) / 3;//ͼ����ȼ������Ϣ��ֵ

	omp_set_num_threads(8);
#pragma omp parallel for

	for (int nI = 0; nI<src->height; nI++)
	{
		for (int nJ = 0; nJ <src->width; nJ++)//���ڿ��ɨ����е�λ���ش���
		{
			for (int nK = 0; nK < 3; nK++)
			{
				nVal = (int)(dContrast * SrcData[(nI*step + nJ) * 3 + nK] + dBright);//��src��Ӧ���صľ���ռ����һ������ֵ�����ڸ���������
				if (nVal < 0)
					nVal = 0;
				if (nVal > 255)
					nVal = 255;
				DstData[(nI*step + nJ) * 3 + nK] = nVal;//�����ͼ��dst���и�ֵ
			}
		}
	}
}
void GetDiffImage(IplImage* src1, IplImage* src2, IplImage* dst, int nThre)
{
	unsigned char* SrcData1 = (unsigned char*)src1->imageData;
	unsigned char* SrcData2 = (unsigned char*)src2->imageData;
	unsigned char* DstData = (unsigned char*)dst->imageData;
	int step = src1->widthStep / sizeof(unsigned char);

	omp_set_num_threads(8);
#pragma omp parallel for //���̲߳���

	for (int nI = 0; nI<src1->height; nI++)
	{
		for (int nJ = 0; nJ <src1->width; nJ++)
		{
			if (SrcData1[nI*step + nJ] - SrcData2[nI*step + nJ]> nThre)
			{
				DstData[nI*step + nJ] = 255;
			}
			else
			{
				DstData[nI*step + nJ] = 0;
			}
		}
	}
}

vector<CvBox2D> ArmorDetect(vector<CvBox2D> vEllipse)
{
	vector<CvBox2D> vRlt;
	CvBox2D Armor;
	int nL, nW;
	double dAngle;
	vRlt.clear();
	if (vEllipse.size() < 2)
		return vRlt;
	for (unsigned int nI = 0; nI < vEllipse.size() - 1; nI++)
	{
		for (unsigned int nJ = nI + 1; nJ < vEllipse.size(); nJ++)
		{
			dAngle = abs(vEllipse[nI].angle - vEllipse[nJ].angle);
			while (dAngle > 180)
				dAngle -= 180;
			if ((dAngle < T_ANGLE_THRE || 180 - dAngle < T_ANGLE_THRE) && 
				abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) < (vEllipse[nI].size.height + vEllipse[nJ].size.height) / T_SIZE_THRE &&
				abs(vEllipse[nI].size.width - vEllipse[nJ].size.width) < (vEllipse[nI].size.width + vEllipse[nJ].size.width) / T_SIZE_THRE)
			{
				Armor.center.x = (vEllipse[nI].center.x + vEllipse[nJ].center.x) / 2;
				Armor.center.y = (vEllipse[nI].center.y + vEllipse[nJ].center.y) / 2;
				Armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle) / 2;
				if (180 - dAngle < T_ANGLE_THRE)
					Armor.angle += 90;
				nL = (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 2;
				nW = sqrt((vEllipse[nI].center.x - vEllipse[nJ].center.x) * (vEllipse[nI].center.x - vEllipse[nJ].center.x) + 
					(vEllipse[nI].center.y - vEllipse[nJ].center.y) * (vEllipse[nI].center.y - vEllipse[nJ].center.y));
				if (nL < nW)
				{
					Armor.size.height = nL;
					Armor.size.width = nW;
				}
				else
				{
					Armor.size.height = nW;
					Armor.size.width = nL;
				}
				vRlt.push_back(Armor);
			}
		}
	}
	return vRlt;
}

void DrawBox(CvBox2D box, IplImage* img)
{
	CvPoint2D32f point[4];
	int i;
	for (i = 0; i<4; i++)
	{
		point[i].x = 0;
		point[i].y = 0;
	}
	cvBoxPoints(box, point); //�����ά���Ӷ��� 
	CvPoint pt[4];
	for (i = 0; i<4; i++)
	{
		pt[i].x = (int)point[i].x;
		pt[i].y = (int)point[i].y;
	}
	cvLine(img, pt[0], pt[1], CV_RGB(0, 0, 255), 2, 8, 0);
	cvLine(img, pt[1], pt[2], CV_RGB(0, 0, 255), 2, 8, 0);
	cvLine(img, pt[2], pt[3], CV_RGB(0, 0, 255), 2, 8, 0);
	cvLine(img, pt[3], pt[0], CV_RGB(0, 0, 255), 2, 8, 0);
}

int main()
{
	CvCapture* pCapture0 = cvCreateFileCapture("RedCar.mp4");
	//CvCapture* pCapture0 = cvCreateCameraCapture(0);
	IplImage* pFrame0 = NULL;

	CvSize pImgSize;
	CvBox2D s;
	vector<CvBox2D> vEllipse;
	vector<CvBox2D> vRlt;
	vector<CvBox2D> vArmor;
	CvScalar sl;
	bool bFlag = false;
	CvSeq *pContour = NULL;
	CvMemStorage *pStorage = cvCreateMemStorage(0);


	pFrame0 = cvQueryFrame(pCapture0);

	pImgSize = cvGetSize(pFrame0);

	IplImage *pRawImg = cvCreateImage(pImgSize, IPL_DEPTH_8U, 3);
	cvShowImage("img",pRawImg);
	cvWaitKey(1);

	IplImage* pGrayImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage* pRImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage* pGImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pBImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pBinary = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pRlt = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);

	CvSeq* lines = NULL;
	CvMemStorage* storage = cvCreateMemStorage(0);
	while (1)
	{
		if (pFrame0)
		{
			cvShowImage("BA_src", pFrame0);
			BrightAdjust(pFrame0, pRawImg, 1, -120);
			cvShowImage("BA_dst", pRawImg);
			

			cvSplit(pRawImg, pBImage, pGImage, pRImage, 0);
			cvShowImage("Diff_src_R", pRImage);
			cvShowImage("Diff_src_G", pGImage);
			GetDiffImage(pRImage, pGImage, pBinary, 25);//��r gͨ��ͼƬ���жԱȶȶԱ��ҳ�������25Ϊ��ֵ���ж�ֵ��
			cvShowImage("Diff_dst", pBinary);
			cvWaitKey(1);

			cvDilate(pBinary, pGrayImage, NULL, 3);
			cvErode(pGrayImage, pRlt, NULL, 1);
			cvFindContours(pRlt, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			for (; pContour != NULL; pContour = pContour->h_next)
			{
				if (pContour->total > 10)
				{
					bFlag = true;
					s = cvFitEllipse2(pContour);
					for (int nI = 0; nI < 5; nI++)
					{
						for (int nJ = 0; nJ < 5; nJ++)
						{
							if (s.center.y - 2 + nJ > 0 && s.center.y - 2 + nJ < pFrame0->height && s.center.x - 2 + nI > 0 && s.center.x - 2 + nI <  pFrame0->width)
							{
								sl = cvGet2D(pFrame0, (int)(s.center.y - 2 + nJ), (int)(s.center.x - 2 + nI));
								if (sl.val[0] < T_LUMI_THRE || sl.val[1] < T_LUMI_THRE || sl.val[2] < T_LUMI_THRE)//Filtrate ellipse contour with high luminance.
									bFlag = false;
							}
						}
					}
					if (bFlag)
					{
						vEllipse.push_back(s);
						//cvEllipseBox(pFrame0, s, CV_RGB(255, 0, 0), 2, 8, 0);
					}
				}
				
			}

			for (unsigned int nI = 0; nI < vEllipse.size(); nI++)
				DrawBox(vEllipse[nI], pFrame0);


			//�Դ����������ͼƬ��������ѭ�����ҳ���Ӧ����������
			vRlt = ArmorDetect(vEllipse);

			for (unsigned int nI = 0; nI < vRlt.size(); nI++)
				DrawBox(vRlt[nI], pFrame0);


			cvShowImage("Raw", pFrame0);
			cvWaitKey(0);
			vEllipse.clear();
			vRlt.clear();
			vArmor.clear();
		}
		pFrame0 = cvQueryFrame(pCapture0);
	}
	cvReleaseCapture(&pCapture0);
	return 0;
}
