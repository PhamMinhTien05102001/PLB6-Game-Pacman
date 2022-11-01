import { Row } from 'antd';
import { observer } from 'mobx-react-lite';
import React, { useEffect } from 'react';
import styled from 'styled-components/macro';
import { Board } from '../../components/Board';
import { DebugView } from './components/DebugView';
import { ExtraLives } from './components/ExtraLives';
import { GameOver } from './components/GameOver';
import { GhostsGameView } from './components/GhostsView';
import { MazeView } from './components/MazeView';
import { PacManView } from './components/PacManView';
import { PillsView } from './components/PillsView';
import { Score } from './components/Score';
import { useStore } from '../../components/StoreContext';
import { useKeyboardActions } from './components/useKeyboardActions';
import { VSpace } from '../../components/Spacer';
import { useGameLoop } from '../../model/useGameLoop';
import WebcamGame from './components/WebcamGame';
import { Direction } from '../../model/Types';
// import { Switch } from 'react-router-dom';
import { useState } from 'react';
import { Switch } from 'antd';
import { Progress } from './components/Progress';

export const GamePage: React.FC = observer(() => {
  const store = useStore();

  useEffect(() => {
    store.resetGame();
    return () => {
      store.game.gamePaused = true;
    };
    // eslint-disable-next-line  react-hooks/exhaustive-deps
  }, []);
  const triggerDirection = (direction: Direction) => {
    store.game.pacMan.nextDirection = direction;
  };
  const triggerChaos = () => {
    store.game.killedGhosts = 0;
    store.game.pacMan.send('ENERGIZER_EATEN');
    for (const ghost of store.game.ghosts) {
      ghost.send('ENERGIZER_EATEN');
    }
  };
  const triggerGamePause = () => {
    store.game.gamePaused = true;
  };
  useGameLoop();
  useKeyboardActions();

  const [hide, setHide] = useState(false);
  const toggleHide = () => {
    setHide(!hide);
  };
  return (
    <>
      <div className="debugbar-toggle-wrap">
        Show Debug Bar
        <Switch onClick={toggleHide} className="debugbar-toggle" />
      </div>
      <Layout data-testid="GamePage">
        <ScoreArea>
          <Row justify="center">
            <Score />
            <Progress />
          </Row>
          <VSpace size="small" />
        </ScoreArea>

        <EmptyArea />

        <BoardArea>
          <Board>
            <MazeView />
            <PillsView />
            <PacManView />
            <GhostsGameView />
            <GameOver />
          </Board>
          <VSpace size="large" />
          <Row justify="center">
            <ExtraLives />
          </Row>
        </BoardArea>

        <WebcamGame
          triggerDirection={triggerDirection}
          triggerGamePause={triggerGamePause}
          triggerChaos={triggerChaos}
        />
      </Layout>
      <div
        className="debugbar-wrap"
        style={{ display: hide ? 'flex' : 'none' }}
      >
        <DebugArea>
          <DebugView />
        </DebugArea>
      </div>
    </>
  );
});

const Layout = styled.div`
  margin-left: 16px;
  margin-right: 16px;

  display: grid;

  @media (min-width: 1280px) {
    grid-template-columns: 1fr 1fr;
    justify-items: center;
  }
`;

const ScoreArea = styled.div``;

const EmptyArea = styled.div``;

const BoardArea = styled.div``;

const DebugArea = styled.div`
  @media (max-width: 1280px) {
    display: none;
  }
`;
