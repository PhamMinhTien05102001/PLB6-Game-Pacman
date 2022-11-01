import { observer } from 'mobx-react-lite';
import { useCallback, useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import styled from 'styled-components';
import axios from 'axios';
import './WebCam.css';
// import { Hands } from '@mediapipe/hands';
// import * as hands from '@mediapipe/hands';
// import { Camera } from '@mediapipe/camera_utils';
import { Point } from 'react-easy-crop/types';
import { API_URL_DEPLOY } from '../../../constant/index';
import Cropper from 'react-easy-crop';
import { Direction } from '../../../model/Types';
import getCroppedImg from '../utils/cropImage';

const videoConstraints = {
  width: 800,
  height: 600,
  facingMode: 'user',
};

const captureConfig = {
  width: 300,
  height: 300,
  timeCapture: 1000,
  acceptThreshold: 2,
  acceptPercent: 90,
};
// interface FingerCors {
//   x: number;
//   y: number;
//   z: number;
//   visibility: any;
// }
type gestureCount = 'Attack' | 'Bottom' | 'Left' | 'Right' | 'Stop' | 'Top';

// const detectHandInBox = (finger: any) => {
//   if (!finger) return false;
//   for (let i = 0; i < finger.length; i++) {
//     if (finger[i].x > 0.3 || finger[i].y > 0.4) {
//       return false;
//     }
//   }
//   return true;
// };

const WebcamGame = observer(
  ({
    triggerDirection,
    triggerGamePause,
    triggerChaos,
  }: {
    triggerDirection: (direction: Direction) => any;
    triggerGamePause: () => any;
    triggerChaos: () => any;
  }) => {
    const [crop, setCrop] = useState<Point>({ x: -800, y: -500 });
    const webcamRef = useRef<any>(null);

    const buttonRef = useRef<HTMLButtonElement>(null);
    const [gesture, setGesture] = useState<string>();
    // const [countHandInBox, setCountHandInBox] = useState<number>(0);
    const gestureCount = {
      Attack: 0,
      Bottom: 0,
      Left: 0,
      Right: 0,
      Stop: 0,
      Top: 0,
    };
    // let camera = null;
    const [imgSrc, setImgSrc] = useState<any>(null);

    const resetGestureCount = () => {
      for (let i in gestureCount) {
        gestureCount[i as gestureCount] = 0;
      }
    };

    const detectGesture = () => {
      for (let i in gestureCount) {
        if (gestureCount[i as gestureCount] >= captureConfig.acceptThreshold) {
          resetGestureCount();
          setGesture(i);
          if (i === 'Bottom') triggerDirection('DOWN');
          if (i === 'Left') triggerDirection('LEFT');
          if (i === 'Right') triggerDirection('RIGHT');
          if (i === 'Top') triggerDirection('UP');
          if (i === 'Attack') triggerChaos();
          if (i === 'Stop') triggerGamePause();
        }
      }
    };

    // const onResults = (results: any) => {
    //   // console.log(results);
    //   setCountHandInBox(val => val + 1);

    //   // if (countHandInBox === 5) {
    //   //   setCountHandInBox(0);
    //   //   console.log('Detect hand in box equal 5 and reset');
    //   // }
    // };

    useEffect(() => {
      const captureInterval = setInterval(() => {
        buttonRef.current?.click();
      }, captureConfig.timeCapture);
      return () => clearInterval(captureInterval);
    }, []);

    // useEffect(() => {
    //   const hands = new Hands({
    //     locateFile: file => {
    //       return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    //     },
    //   });
    //   hands.setOptions({
    //     maxNumHands: 1,
    //     modelComplexity: 1,
    //     minDetectionConfidence: 0.5,
    //     minTrackingConfidence: 0.5,
    //   });

    //   hands.onResults(hand => {
    //     // console.log(detectHandInBox(hand.multiHandLandmarks[0]));
    //     if (detectHandInBox(hand.multiHandLandmarks[0])) {
    //       onResults(hand);
    //     }
    //   });

    //   if (
    //     typeof webcamRef.current !== 'undefined' &&
    //     webcamRef.current !== null
    //   ) {
    //     camera = new Camera(webcamRef.current.video, {
    //       onFrame: async () => {
    //         await hands.send({ image: webcamRef.current.video });
    //       },
    //       width: 800,
    //       height: 600,
    //     });
    //     camera.start();
    //   }
    // }, []);

    const capture = useCallback(async () => {
      const imageSrc = webcamRef.current.getScreenshot({});
      setImgSrc(imageSrc);
      const imgCrop = await getCroppedImg(imageSrc, {
        width: captureConfig.width,
        height: captureConfig.height,
        x: videoConstraints.width - captureConfig.width,
        y: videoConstraints.height - captureConfig.height,
      });

      const form = new FormData();
      form.append('imageFile', imgCrop);
      // console.log(imgCrop);
      axios({
        method: 'post',
        url: API_URL_DEPLOY,
        data: form,
        headers: {
          'Content-Type': 'multipart/form-data',
          'Access-Control-Allow-Origin': '*',
        },
      })
        .then(function(response: any) {
          //handle success

          // console.log(response.data);
          if (response.data['Percent'] >= captureConfig.acceptPercent) {
            // console.log(response.data['Class Name'], 'Plus one');
            gestureCount[response.data['Class Name'] as gestureCount] += 1;
          }

          detectGesture();
          // setGesture(response.data['Class Name']);
        })
        .catch(function(response) {
          console.log(response);
        });
    }, [webcamRef, setImgSrc]);

    return (
      <Layout>
        <div className="my-video" style={{ position: 'relative' }}>
          <CaptureFrame />
          <Webcam
            audio={false}
            width={800}
            height={600}
            ref={webcamRef}
            screenshotFormat="image/png"
            mirrored={true}
            videoConstraints={videoConstraints}
          />
        </div>
        <p style={{ color: 'white' }}>{gesture}</p>
        <button onClick={capture} style={{ display: 'none' }} ref={buttonRef}>
          Capture Image
        </button>
        {imgSrc && (
          <Cropper
            image={imgSrc}
            crop={crop}
            cropSize={{
              width: captureConfig.width,
              height: captureConfig.height,
            }}
            aspect={1}
            onCropChange={setCrop}
          />
        )}
      </Layout>
    );
  }
);

const CaptureFrame = styled.div`
  position: absolute;
  width: ${captureConfig.width}px;
  height: ${captureConfig.height}px;
  border: 3px solid salmon;
  right: 0px;
  bottom: 3px;
  z-index: 20;
`;

const Layout = styled.div`
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
`;
export default WebcamGame;
