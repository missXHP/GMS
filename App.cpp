#include<opencv2/opencv.hpp>
#include<opencv2/stitching.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/stitching/detail/matchers.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv_modules.hpp"

#include<Windows.h>

using namespace std;
using namespace cv;

class computedata
{
public:

	Mat imag_count(Mat frame , Mat frame_prev)
	{
		auto bmf = BFMatcher::create(NORM_HAMMING);
		auto orb = ORB::create(5000);
		orb->setEdgeThreshold(0);
		orb->setFastThreshold(0);
		orb->detectAndCompute(frame, Mat(), kp, desc);
		orb->detectAndCompute(frame_prev, Mat(), kp_prev, desc_prev);
		if (!desc_prev.empty())
			bmf->match(desc, desc_prev, matches);
		xfeatures2d::matchGMS(
			frame.size(), frame_prev.size(),
			kp, kp_prev,
			matches, matches,
			false, false,
			10);
		Mat canvas(Size(frame.cols * 2, frame.rows), CV_8UC3);
		drawMatches(frame, kp, frame_prev, kp_prev, matches, canvas,
			Scalar(0, 32, 64), Scalar(128, 255, 0), vector<char>(), DrawMatchesFlags::DRAW_OVER_OUTIMG);
		Mat view;
		cv::hconcat(frame, frame_prev, view);
		addWeighted(view, 1, canvas, -1, 0, view);
		pyrDown(view, view);
		return view;
	};
	~computedata();

public:
	vector<KeyPoint>	 kp, kp_prev;
	Mat				 desc, desc_prev;
	vector<DMatch>		 matches;
	
};

computedata::~computedata()
{
}

Mat colorheat(Mat& frame, int mode = 2) {
	Mat out(frame.size(), CV_8U);
	size_t sz = frame.size().area();
	switch (mode)
	{
	case 0:	// maxcolor channel
	{
		for (size_t i = 0; i < sz; i++) {
			auto i_ = i * 3;
			auto min_ = min(frame.data[i_], min(frame.data[i_ + 1], frame.data[i_ + 2]));
			auto max_ = max(frame.data[i_], max(frame.data[i_ + 1], frame.data[i_ + 2]));
			out.data[i] = max_ - min_;
		}
		break;
	}
	case 1:	// max channel
	{
		for (size_t i = 0; i < sz; i++) {
			auto i_ = i * 3;
			auto max_ = max(frame.data[i_], max(frame.data[i_ + 1], frame.data[i_ + 2]));
			out.data[i] = max_;
		}
		break;
	}
	case 2:	// dark channel
	{
		for (size_t i = 0; i < sz; i++) {
			auto i_ = i * 3;
			auto min_ = min(frame.data[i_], min(frame.data[i_ + 1], frame.data[i_ + 2]));
			out.data[i] = min_;
		}
		break;
	}
	case 3:	// mean greay
	{
		for (size_t i = 0; i < sz; i++) {
			auto i_ = i * 3;
			out.data[i] = (frame.data[i_] + frame.data[i_ + 1] + frame.data[i_ + 2]) / 3;
		}
		break;
	}

	case 4: //grdiff
	{
		for (size_t i = 0; i < sz; i++) {
			auto i_ = i * 3;
			out.data[i] = abs(frame.data[i_ + 1] - frame.data[i_ + 2]);
		}
		break;
	}
	default:
		break;
	}
	return out;
}
void initcapture(VideoCapture cap,VideoCapture cap2)
{
	
	cap2.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap2.set(CAP_PROP_FRAME_HEIGHT, 720);
	cap2.set(CAP_PROP_FOURCC, 'GPJM');

	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);
	cap.set(CAP_PROP_FOURCC, 'GPJM');
	Mat frame, frame_prev;
	cap >> frame;
	cap2 >> frame_prev;
}




class ORBGMSRequest
{
public:
	void init() {
		thread([this]() {
			vector<KeyPoint>	 kp1, kp2;
			Mat				 desc1, desc2;
			vector<DMatch>		 matches;
			auto bmf = BFMatcher::create(NORM_HAMMING);
			auto orb = ORB::create(500);
			orb->setEdgeThreshold(0);
			orb->setFastThreshold(0);
			spin_status = SPIN_IDLE;
			while (1)
			{
				for (; spin_status != SPIN_BUSY; Sleep(1));

				//cout << i1.size() << i2.size() << endl;
				orb->detectAndCompute(i1, Mat(), kp1, desc1);
				orb->detectAndCompute(i2, Mat(), kp2, desc2);
				if (!desc2.empty())
					bmf->match(desc1, desc2, matches);
				if(kp1.size() && kp2.size())
					xfeatures2d::matchGMS(
						i1.size(), i2.size(),
						kp1, kp2,
						matches, matches,
						false, false,
						10);
				ret.p1.clear();
				ret.p1.reserve(kp1.size());
				for (auto& m:matches)
					ret.p1.push_back(kp1[m.queryIdx].pt);
				ret.p2.clear();
				ret.p2.reserve(kp2.size());
				for (auto& m : matches)
					ret.p2.push_back(kp2[m.trainIdx].pt);

				spin_status = SPIN_IDLE;
			}
		}).detach();
		for (; spin_status != SPIN_IDLE; Sleep(1));
	}
	struct kp_pairs{
		vector<Point> p1, p2;
	};
	kp_pairs submit(Mat& f1, Mat& f2) {
		//cout << "s1" << endl;
		for (; spin_status != SPIN_IDLE; Sleep(1));
		//cout << "s2" << endl;
		f1.copyTo(i1);				f2.copyTo(i2);
		spin_status = SPIN_BUSY;
		return ret;
	}
	bool ready = false;
	kp_pairs ret;
private:

	Mat i1, i2; 

	enum
	{
		SPIN_IDLE,
		SPIN_BUSY
	} spin_status = SPIN_BUSY;



};

class ORBGMS
{
public:
	ORBGMS(size_t n_threads) :req(n_threads), c1(n_threads), c2(n_threads) {
		for (auto& r : req)r.init();
	
	}
	struct MatchRslt {
		Mat f1, f2;
		ORBGMSRequest::kp_pairs matched;
	};
	//将图片上传处理并更新图片信息
	MatchRslt spinOnce(Mat& f1, Mat& f2) {
		auto i = idx();
		MatchRslt ret;

		ret.matched = req[i].submit(f1, f2);
		ret.f1 = c1[i];
		ret.f2 = c2[i];
		//cout <<i<<" "<< ret.f2.size() << endl;
		f1.copyTo(c1[i]);
		f2.copyTo(c2[i]);
		return ret;
	}
	
	bool initial_request(queue<Mat>& f1, queue<Mat>& f2) {
		if (f1.size() < req.size() || f2.size() < req.size())return false;
		for (size_t i = 0; i < req.size(); i++){
			req[i].submit(f1.front(), f2.front());
			f1.front().copyTo(c1[i]);
			f2.front().copyTo(c2[i]);
			f1.pop();
			f2.pop();
		}
		return true;
	}
	ORBGMSRequest::kp_pairs flush() {
		return req[idx()].ret;
	}
private:
	size_t idx() {
		auto ret = _idx;
		_idx++;
		if (_idx >= req.size())
			_idx = 0;
		return ret;
	}
	size_t _idx = 0;
	vector<ORBGMSRequest> req;
	vector<Mat> c1,c2;
};

int main()
{
	
	String file= "C:\\Users\\mc_xia\\Desktop\\gms_orb_matching\\gms_orb_matching\\date\\";
	fstream infile(file + "record.txt", ios::app);
	ORBGMS orbgms(2);
	queue<Mat> frame[2];
	thread([&]() {
		/*vector<VideoCapture> cap(2);
		size_t cam_idx = 0;
		for (auto&c:cap)
			c.open(700 + cam_idx++);*/
		//VideoWriter out;
		//out.open("video.mp4", 'GPJM', 30.00, Size(1280, 720), true);
		
		VideoCapture cap1("C:\\Users\\mc_xia\\Desktop\\gms_orb_matching\\gms_orb_matching\\date\\video1.mp4");
		VideoCapture cap2("C:\\Users\\mc_xia\\Desktop\\gms_orb_matching\\gms_orb_matching\\date\\video2.mp4");
		while (1)
		{
			/*for (size_t i = 0; i < 2; i++) {
				try
				{
					if (frame[i].size() > 100) continue;
					Mat _frame;
					cap[i] >> _frame;
					if (_frame.empty())continue;
					frame[i].push(_frame.clone());
					imshow("camera " + to_string(i), _frame);
					waitKey(1);
					
				}
				catch (const std::exception&e)
				{
				//	cout << e.what() << endl;
				}
			}*/
			
				for (size_t i = 0; i < 2; i++)
				{
					if (frame[i].size() > 100) continue;
				}
				Mat _frame;
				cap1 >> _frame;
				if (_frame.empty())continue;
				pyrDown(_frame, _frame, Size(_frame.cols / 2, _frame.rows / 2));
				frame[0].push(_frame.clone());
				imshow("camera1 ", _frame.clone());

				cap2 >> _frame;
				if (_frame.empty())continue;
				pyrDown(_frame, _frame, Size(_frame.cols / 2, _frame.rows / 2));
				frame[1].push(_frame.clone());
				imshow("camera2 ", _frame.clone());
				//Sleep(1);
				waitKey(1);
		}
	}
	).detach();


	for (; !orbgms.initial_request(frame[0], frame[1]); Sleep(1));
	for (; waitKey(1) != 27; [&]() {for (Sleep(1); frame[0].empty() || frame[1].empty(); Sleep(1)); })
	{
		//cout << "---"<<frame[0].front().size() << frame[1].front().size() << endl;
		auto ret = orbgms.spinOnce(frame[0].front(), frame[1].front());
		frame[0].pop(); frame[1].pop();
		Mat view;
		hconcat(ret.f1, ret.f2, view);
		Mat canvas = Mat::zeros(view.size(), CV_8UC3);
		cout << "Match:" << ret.matched.p1.size() << endl;
		for (size_t i = 0; i < ret.matched.p1.size(); i++) {
			
			auto& p1 = ret.matched.p1[i];
			auto& p2 = ret.matched.p2[i];
			line(canvas, p1, p2 + Point(view.cols/2,0), Scalar(0, 128, 64));
			infile << p1 << "," << p2 << endl;
		}
		addWeighted(view, 1, canvas, -1, 0, view);
		pyrDown(view, view, Size(view.cols / 2, view.rows / 2));
		imshow("view", view);

	}
}

