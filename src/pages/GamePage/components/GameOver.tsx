import { observer } from 'mobx-react-lite';
import React, { FC } from 'react';
import './GameOver.css';
import { useGame } from '../../../components/StoreContext';
import { Message } from './Message';
import { TotalPacManDyingAnimationLength } from '../../../model/pacManDyingPhase';
import styled from 'styled-components';

export const TOTAL_TIME_TO_GAME_OVER_MESSAGE = TotalPacManDyingAnimationLength;

export const GameOver: FC<{ className?: string }> = observer(
  ({ className }) => {
    const game = useGame();

    const { pacMan } = game;
    const gameOverMessageVisible =
      game.gameOver && pacMan.timeSinceDeath >= TOTAL_TIME_TO_GAME_OVER_MESSAGE;

    return gameOverMessageVisible ? (
      <>
        <Message text="Game Over" />
        <Button
          onClick={() => {
            window.location.reload();
          }}
        >
          Reset Game
        </Button>
      </>
    ) : null;
  }
);
const Button = styled.button`
  font-family: Joystix;
  font-size: 14px;
  background: yellow;
  color: white;
  cursor: pointer;
  position: absolute;
  left: 170px;
  top: 370px;
  width: 220px;
  text-align: center;
`;
