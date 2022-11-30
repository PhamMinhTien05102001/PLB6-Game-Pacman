/* eslint-disable react/display-name */
import { Button, Card, Col, Radio, Row, Switch, Typography } from 'antd';
import { observer } from 'mobx-react-lite';
import React, { useState } from 'react';
import styled from 'styled-components/macro';
import { useGame, useStore } from '../../../components/StoreContext';
import { action } from 'mobx';
import { RadioChangeEvent } from 'antd/lib/radio';

const { Text } = Typography;

const options = [
  { label: 1, value: 1 },
  { label: 2, value: 2 },
];

export const GameDebugView = observer<{ className?: string }>(
  ({ className }) => {
    const store = useStore();
    const game = useGame();

    const [value, setValue] = useState<any>(game.speed);

    const onSpeedChange = ({ target: { value } }: RadioChangeEvent) => {
      localStorage.setItem('GAMESPEED', value);
      setValue(value);
      window.location.reload();
    };
    return (
      <Layout className="PacManDebugView">
        <Card title="Game" size="small" bordered={false}>
          <Row>
            <Col flex="0 0 56px">
              <Switch
                checked={store.debugState.gameViewOptions.hitBox}
                onChange={action(
                  checked => (store.debugState.gameViewOptions.hitBox = checked)
                )}
              />
            </Col>
            <Col flex="0 0 auto">
              <Text>Show Hit Boxes</Text>
            </Col>
            <Col flex="0 0 48px"></Col>

            <Col flex="0 0 56px">
              <Switch
                checked={game.gamePaused}
                onChange={checked => {
                  game.gamePaused = checked;
                }}
              />
            </Col>
            <Col flex="0 0 auto">
              <Text>Paused</Text>
            </Col>
            <Col flex="0 0 48px"></Col>

            <ButtonStyled size="small" onClick={store.resetGame} shape="round">
              Restart
            </ButtonStyled>
          </Row>
          <Row style={{ marginTop: '20px' }} align="middle" gutter={12}>
            <Col flex="0 0 auto">
              <Radio.Group
                options={options}
                onChange={onSpeedChange}
                value={value}
                optionType="button"
              />
            </Col>
            <Col flex="0 0 auto">
              <Text>Game speed</Text>
            </Col>

            <Col flex="0 0 auto">
              <Switch
                checked={game.showAcc}
                onChange={checked => {
                  game.showAcc = checked;
                }}
              />
            </Col>
            <Col flex="0 0 auto">
              <Text>Show Gesture Acc</Text>
            </Col>
          </Row>
        </Card>
      </Layout>
    );
  }
);

const Layout = styled.div``;

const ButtonStyled = styled(Button)`
  min-width: 80px;
`;
