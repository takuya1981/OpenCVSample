using OpenCvSharp;
using OpenCvSharp.CPlusPlus;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using System.IO;

namespace WebApplication2.Controllers
{
    public class HomeController : Controller
    {
        public ActionResult Index()
        {
            return View();
        }

        public ActionResult About()
        {
            ViewBag.Message = "Your application description page.";

            return View();
        }

        public ActionResult Contact()
        {
            ViewBag.Message = "Your contact page.";

            return View();
        }

        public ActionResult Canny(HttpPostedFileBase imageData)
        {
            if (imageData != null)
            {
                using (var image = IplImage.FromStream(imageData.InputStream, LoadMode.Color))
                {
                    using (var grayImage = new IplImage(image.Size, BitDepth.U8, 1))
                    using (var cannyImage = new IplImage(image.Size, BitDepth.U8, 1))
                    {
                        Cv.CvtColor(image, grayImage, ColorConversion.BgrToGray);
                        Cv.Canny(grayImage, cannyImage, 60, 180);

                        byte[] cannyBytes = cannyImage.ToBytes(".png");
                        string base64 = Convert.ToBase64String(cannyBytes);
                        ViewBag.Base64Image = base64;

                        byte[] originalBytes = image.ToBytes(".png");
                        string base64Org = Convert.ToBase64String(originalBytes);
                        ViewBag.Base64OrgImage = base64Org;

                        byte[] grayBytes = grayImage.ToBytes(".png");
                        string base64Gray = Convert.ToBase64String(grayBytes);
                        ViewBag.Base64GrayImage = base64Gray;
                    }
                }
            }

            return View();
        }

        public ActionResult DetectFace(HttpPostedFileBase imageData)
        {
            try
            {
                if (imageData == null) { throw new ArgumentException("File is not exist."); }
                
                using (var img = Mat.FromStream(imageData.InputStream, LoadMode.Color))
                {
                    var ExecutingAssemblyPath = new Uri(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().CodeBase.ToString())).LocalPath;
                    double scale = 2.0;
                    using (var gray = new Mat())
                    using (var smallImg = new Mat((int)(img.Rows / scale), (int)(img.Cols / scale), MatType.CV_8UC1))
                    {
                        Cv2.CvtColor(img, gray, ColorConversion.BgrToGray);
                        Cv2.Resize(gray, smallImg, smallImg.Size(), 0, 0, Interpolation.Linear);
                        Cv2.EqualizeHist(smallImg, smallImg);

                        byte[] imgBytes = img.ToBytes(".png");
                        string base64Img = Convert.ToBase64String(imgBytes);
                        ViewBag.Base64Img = base64Img;
                       
                        var obj = new CascadeClassifier();
                        var cascadeFilePath = Path.Combine(ExecutingAssemblyPath, "Content\\haarcascade_frontalface_alt.xml");
                        if (!obj.Load(cascadeFilePath)) { throw new InvalidOperationException("Failed to load classifier file."); }

                        var rects = obj.DetectMultiScale(smallImg);

                        var nestedObj = new CascadeClassifier();
                        var nestedCascadeFilePath = Path.Combine(ExecutingAssemblyPath, "Content\\haarcascade_eye.xml");
                        //var nestedCascadeFilePath = Path.Combine(ExecutingAssemblyPath, "Content\\haarcascade_eye_tree_eyeglasses.xml");
                        if (!nestedObj.Load(nestedCascadeFilePath)) { throw new InvalidOperationException("Failed to load classifier file."); }
                        foreach(var rect in rects)
                        {
                            Point faceCenter;
                            faceCenter.X = (int)((rect.X + rect.Width * 0.5) * scale);
                            faceCenter.Y = (int)((rect.Y + rect.Height * 0.5) * scale);
                            int faceRadius = (int)((rect.Width + rect.Height) * 0.25 * scale);
                            Cv2.Circle(img, faceCenter, faceRadius, new Scalar(80, 80, 255), 3, LineType.Link8, 0);

                            Mat smallImgROI = new Mat(smallImg, rect);
                            var nestedRects = nestedObj.DetectMultiScale(smallImgROI);

                            foreach(var nestedRect in nestedRects)
                            {
                                Point center;
                                center.X = (int)((rect.X + nestedRect.X + nestedRect.Width * 0.5) * scale);
                                center.Y = (int)((rect.Y +  nestedRect.Y + nestedRect.Height * 0.5) * scale);
                                int radius = (int)((nestedRect.Width + nestedRect.Height) * 0.25 * scale);
                                Cv2.Circle(img, center, radius, new Scalar(80, 255, 80), 3, LineType.Link8, 0);
                            }
                        }
                        
                        byte[] resultBytes = img.ToBytes(".png");
                        string base64Result = Convert.ToBase64String(resultBytes);
                        ViewBag.Base64OrgResult = base64Result;

                        byte[] grayBytes = gray.ToBytes(".png");
                        string base64Gray = Convert.ToBase64String(grayBytes);
                        ViewBag.Base64Gray = base64Gray;

                        byte[] smallBytes = smallImg.ToBytes(".png");
                        string base64Small = Convert.ToBase64String(smallBytes);
                        ViewBag.Base64Small = base64Small;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return View();
        }

        public ActionResult DetectHuman(HttpPostedFileBase imageData)
        {
            try
            {
                if (imageData == null) { throw new ArgumentException("File is not exist."); }

                using (var img = Mat.FromStream(imageData.InputStream, LoadMode.Color))
                {
                    var ExecutingAssemblyPath = new Uri(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().CodeBase.ToString())).LocalPath;
                    double scale = 2.0;
                    using (var gray = new Mat())
                    using (var smallImg = new Mat((int)(img.Rows / scale), (int)(img.Cols / scale), MatType.CV_8UC1))
                    {
                        byte[] imgBytes = img.ToBytes(".png");
                        string base64Img = Convert.ToBase64String(imgBytes);
                        ViewBag.Base64Img = base64Img;

                        var hog = new HOGDescriptor();
                        hog.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());
                        var rects = hog.DetectMultiScale(img);

                        foreach (var rect in rects)
                        {
                            var r = rect;
                            r.X += Cv.Round(rect.Width * 0.1);
                            r.Width = Cv.Round(rect.Width * 0.8);
                            r.Y += Cv.Round(rect.Height * 0.1);
                            r.Height = Cv.Round(rect.Height * 0.8);
                            Cv2.Rectangle(img, r, new Scalar(0, 255, 0), 3);
                        }

                        byte[] resultBytes = img.ToBytes(".png");
                        string base64Result = Convert.ToBase64String(resultBytes);
                        ViewBag.Base64OrgResult = base64Result;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return View();
        }

        public ActionResult HoughTransform(HttpPostedFileBase imageData)
        {
            try
            {
                if (imageData == null) { throw new ArgumentException("File is not exist."); }

                using (var img = Mat.FromStream(imageData.InputStream, LoadMode.Color))
                {
                    using(var gray = new Mat())
                    using(var workLine = new Mat())
                    using(var workCircle = new Mat())
                    using(var dstStandardHough = img.Clone())
                    using(var dstProbabilistic = img.Clone())
                    using (var dstCircle = img.Clone())
                    {
                        Cv2.CvtColor(img, gray, ColorConversion.BgrToGray);
                        Cv2.Canny(gray, workLine, 50, 200);

                        byte[] imgBytes = img.ToBytes(".png");
                        string base64Img = Convert.ToBase64String(imgBytes);
                        ViewBag.Base64Img = base64Img;

                        #region Standard Hough Transform
                        {
                            var lines = Cv2.HoughLines(workLine, 1, Cv.PI / 180, 200);
                            foreach (var line in lines)
                            {
                                Point pt1, pt2;
                                double a = Math.Cos(line.Theta), b = Math.Sin(line.Theta);
                                double x0 = a * line.Rho, y0 = b * line.Rho;
                                pt1.X = (int)(x0 + 1000 * (-b));
                                pt1.Y = (int)(y0 + 1000 * (a));
                                pt2.X = (int)(x0 - 1000 * (-b));
                                pt2.Y = (int)(y0 - 1000 * (a));
                                Cv2.Line(dstStandardHough, pt1, pt2, new Scalar(0, 0, 255), 3);
                            }
                        }
                        #endregion

                        #region Probabilistic Hough Transform
                        {
                            var lines = Cv2.HoughLinesP(workLine, 1, Cv.PI / 180, 200);
                            foreach (var line in lines)
                            {
                                Cv2.Line(dstProbabilistic, line.P1, line.P2, new Scalar(0, 0, 255), 3);
                            }
                        }
                        #endregion

                        #region Circle
                        {
                            Cv2.GaussianBlur(gray, workCircle, new Size(11, 11), 2, 2);
                            var circles = Cv2.HoughCircles(workCircle, HoughCirclesMethod.Gradient, 1, 100, 20, 50);
                            foreach(var circle in circles)
                            {
                                Point center  = new Point(circle.Center.X, circle.Center.Y);
                                Cv2.Circle(dstCircle, center, (int)(circle.Radius), new Scalar(0, 0, 255), 2);
                            }
                        }
                        #endregion
                        byte[] grayBytes = workLine.ToBytes(".png");
                        string base64Gray = Convert.ToBase64String(grayBytes);
                        ViewBag.Base64Gray = base64Gray;

                        byte[] resultSBytes = dstStandardHough.ToBytes(".png");
                        string base64ResultS = Convert.ToBase64String(resultSBytes);
                        ViewBag.Base64OrgResultS = base64ResultS;

                        byte[] resultPBytes = dstProbabilistic.ToBytes(".png");
                        string base64ResultP = Convert.ToBase64String(resultPBytes);
                        ViewBag.Base64OrgResultP = base64ResultP;

                        byte[] resultCBytes = dstCircle.ToBytes(".png");
                        string base64ResultC = Convert.ToBase64String(resultCBytes);
                        ViewBag.Base64OrgResultC = base64ResultC;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return View();
        }
        public ActionResult Histgram(HttpPostedFileBase imageData)
        {
            try
            {
                if (imageData == null) { throw new ArgumentException("File is not exist."); }

                using (var img = Mat.FromStream(imageData.InputStream, LoadMode.Color))
                {
                    int chWidth = 260;
                    int chHeight = 200;
                    using (var mask = new Mat())
                    using (var hist = new Mat())
                    using (var histImg = new Mat(new Size(chWidth, chHeight), MatType.CV_8UC3, Scalar.All(255)))
                    {
                        Mat[] images = new Mat[] { img };
                        int[] channels = new int[] { 0 };
                        int[] hdims = new int[] { 256 };
                        float[] hranges = new float[] { 0, 256 };
                        float[][] ranges = new float[][] { hranges };
                        Cv2.CalcHist(images, channels, mask, hist, 1, hdims, ranges);

                        double minVal, maxVal;
                        Cv2.MinMaxLoc(hist, out minVal, out maxVal);

                        for (int j = 0; j < hdims[0]; ++j)
                        {
                            int binW = (int)((double)chWidth / hdims[0]);
                            Cv2.Rectangle(histImg, new Point(j * binW, histImg.Rows), new Point((j + 1) * binW, histImg.Rows - (int)(hist.At<float>(j) * (maxVal != 0 ? chHeight/ maxVal : 0.0))), Scalar.All(100));
                        }

                        byte[] imgBytes = img.ToBytes(".png");
                        string base64Img = Convert.ToBase64String(imgBytes);
                        ViewBag.Base64Img = base64Img;

                        byte[] result = histImg.ToBytes(".png");
                        string base64Result = Convert.ToBase64String(result);
                        ViewBag.Base64Result = base64Result;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return View();
        }

        public ActionResult MatchTemplate()
        {
            try
            {
                if (Request.Files.Count != 2) { throw new ArgumentException("File is not uploaded."); }

                HttpPostedFileBase postedImageData = Request.Files[0], postedTemplate = Request.Files[1];

                using (var img = Mat.FromStream(postedImageData.InputStream, LoadMode.Color))
                using (var template = Mat.FromStream(postedTemplate.InputStream, LoadMode.Color))
                {
                    using (var result = new Mat())
                    {
                        byte[] imgBytes = img.ToBytes(".png");
                        string base64Img = Convert.ToBase64String(imgBytes);
                        ViewBag.Base64Img = base64Img;

                        byte[] templateBytes = template.ToBytes(".png");
                        string base64Template = Convert.ToBase64String(templateBytes);
                        ViewBag.Base64Template = base64Template;


                        Cv2.MatchTemplate(img, template, result, MatchTemplateMethod.CCoeffNormed);

                        var roi = new Rect(0,0, template.Cols, template.Rows);
                        Point minPoint, maxPoint;
                        double minVal, maxVal;
                        Cv2.MinMaxLoc(result, out minVal, out maxVal, out minPoint, out maxPoint);
                        roi.X = maxPoint.X;
                        roi.Y = maxPoint.Y;
                        Cv2.Rectangle(img, roi, new Scalar(0, 0, 255), 3);


                        byte[] searchBytes = result.ToBytes(".png");
                        string base64Search = Convert.ToBase64String(searchBytes);
                        ViewBag.Base64Search = base64Search;

                        byte[] resultBytes = img.ToBytes(".png");
                        string base64Result = Convert.ToBase64String(resultBytes);
                        ViewBag.Base64Result = base64Result;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return View();
        }

        public ActionResult FourierTransform(HttpPostedFileBase imageData)
        {
            try
            {
                if (imageData == null) { throw new ArgumentException("File is not exist."); }

                using (var img = Mat.FromStream(imageData.InputStream, LoadMode.Color))
                using(var padded = new Mat())
                {
                    using (var result = new Mat())
                    {
                        byte[] imgBytes = img.ToBytes(".png");
                        string base64Img = Convert.ToBase64String(imgBytes);
                        ViewBag.Base64Img = base64Img;

                        int m = Cv2.GetOptimalDFTSize(img.Rows);
                        int n = Cv2.GetOptimalDFTSize(img.Cols);
                        Cv2.CopyMakeBorder(img, padded, 0, m - img.Rows, 0, n - img.Cols, BorderType.Constant, Scalar.All(0));

                        var planes = new Mat[]{};
                        var complexI = new Mat();
                        Cv2.Merge(planes, complexI);

                        Cv2.Dft(complexI, complexI);

                        // Compute the magnitude
                        planes = Cv2.Split(complexI);
                        var magI = new Mat();
                        Cv2.Magnitude(planes[0], planes[1], magI);

                        magI += Scalar.All(1);
                        Cv2.Log(magI, magI);

                        //magI = magI(Rect(0, 0, magI.Cols & -2, magI.Rows & -2));

                        int cx = magI.Cols / 2;
                        int cy = magI.Rows / 2;

                        var q0 = new Mat(magI, new Rect(0, 0, cx, cy));
                        var q1 = new Mat(magI, new Rect(cx, 0, cx, cy));
                        var q2 = new Mat(magI, new Rect(0, cy, cx, cy));
                        var q3 = new Mat(magI, new Rect(cx, cy, cx, cy));

                        var tmp = new Mat();
                        q0.CopyTo(tmp);
                        q3.CopyTo(q0);
                        tmp.CopyTo(q3);

                        q1.CopyTo(tmp);
                        q2.CopyTo(q1);
                        tmp.CopyTo(q2);

                        Cv2.Normalize(magI, magI, 0, 1, NormType.MinMax);

                        byte[] resultBytes = magI.ToBytes(".png");
                        string base64Result = Convert.ToBase64String(resultBytes);
                        ViewBag.Base64Result = base64Result;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            return View();
        }
    }
}