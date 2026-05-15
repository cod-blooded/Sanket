import { useEffect, useRef } from "react";
import { Animated, Easing, StyleSheet, View } from "react-native";
import Svg, {
    Circle,
    Defs,
    G,
    LinearGradient as SvgLinearGradient,
    Path,
    Stop,
    Text,
} from "react-native-svg";

const AnimatedG = Animated.createAnimatedComponent(G);

type SanketAnimatedLogoProps = {
    width?: number;
    height?: number;
};

export function SanketAnimatedLogo({ width = 130, height = 54 }: SanketAnimatedLogoProps) {
    const handOpacity = useRef(new Animated.Value(0)).current;
    const bubbleOpacity = useRef(new Animated.Value(0)).current;
    const dotOpacity = useRef(new Animated.Value(0)).current;
    const wordOpacity = useRef(new Animated.Value(0)).current;
    const pulse = useRef(new Animated.Value(0)).current;

    useEffect(() => {
        Animated.sequence([
            Animated.timing(handOpacity, {
                toValue: 1,
                duration: 360,
                easing: Easing.out(Easing.cubic),
                useNativeDriver: false,
            }),
            Animated.parallel([
                Animated.timing(bubbleOpacity, {
                    toValue: 1,
                    duration: 320,
                    easing: Easing.out(Easing.cubic),
                    useNativeDriver: false,
                }),
                Animated.timing(dotOpacity, {
                    toValue: 1,
                    duration: 480,
                    easing: Easing.out(Easing.cubic),
                    useNativeDriver: false,
                }),
            ]),
            Animated.timing(wordOpacity, {
                toValue: 1,
                duration: 360,
                easing: Easing.out(Easing.cubic),
                useNativeDriver: false,
            }),
        ]).start();

        Animated.loop(
            Animated.sequence([
                Animated.timing(pulse, {
                    toValue: 1,
                    duration: 1200,
                    easing: Easing.inOut(Easing.cubic),
                    useNativeDriver: true,
                }),
                Animated.timing(pulse, {
                    toValue: 0,
                    duration: 1200,
                    easing: Easing.inOut(Easing.cubic),
                    useNativeDriver: true,
                }),
            ]),
        ).start();
    }, [bubbleOpacity, dotOpacity, handOpacity, pulse, wordOpacity]);

    const scale = pulse.interpolate({
        inputRange: [0, 1],
        outputRange: [1, 1.025],
    });

    return (
        <View style={[styles.wrap, { width, height }]}>
            <Animated.View style={[styles.logoMotion, { transform: [{ scale }] }]}>
                <Svg width={width} height={height} viewBox="0 0 1254 1254">
                    <Defs>
                        <SvgLinearGradient
                            id="handGrad"
                            x1="368"
                            y1="186"
                            x2="620"
                            y2="742"
                            gradientUnits="userSpaceOnUse"
                        >
                            <Stop offset="0%" stopColor="#1598EA" />
                            <Stop offset="38%" stopColor="#0C73C2" />
                            <Stop offset="100%" stopColor="#003E88" />
                        </SvgLinearGradient>
                        <SvgLinearGradient
                            id="aquaGrad"
                            x1="516"
                            y1="660"
                            x2="904"
                            y2="232"
                            gradientUnits="userSpaceOnUse"
                        >
                            <Stop offset="0%" stopColor="#25C8C8" />
                            <Stop offset="100%" stopColor="#11BAB4" />
                        </SvgLinearGradient>
                    </Defs>

                    <AnimatedG opacity={handOpacity}>
                        <Path
                            d="M 410.0 241.0 L 402.0 249.0 L 383.0 326.0 L 375.0 373.0 L 370.0 423.0 L 370.0 489.0 L 372.0 510.0 L 380.0 550.0 L 390.0 580.0 L 407.0 614.0 L 426.0 642.0 L 450.0 669.0 L 477.0 691.0 L 496.0 703.0 L 534.0 720.0 L 564.0 728.0 L 596.0 731.0 L 596.0 729.0 L 577.0 720.0 L 551.0 702.0 L 530.0 681.0 L 520.0 666.0 L 514.0 651.0 L 512.0 642.0 L 512.0 624.0 L 515.0 612.0 L 527.0 592.0 L 585.0 538.0 L 607.0 509.0 L 622.0 476.0 L 633.0 441.0 L 632.0 434.0 L 628.0 429.0 L 622.0 426.0 L 612.0 426.0 L 600.0 431.0 L 588.0 443.0 L 574.0 470.0 L 562.0 484.0 L 544.0 495.0 L 532.0 498.0 L 514.0 495.0 L 501.0 484.0 L 494.0 469.0 L 492.0 455.0 L 495.0 440.0 L 501.0 429.0 L 510.0 420.0 L 541.0 397.0 L 568.0 406.0 L 592.0 421.0 L 603.0 423.0 L 616.0 421.0 L 623.0 415.0 L 626.0 408.0 L 624.0 397.0 L 612.0 383.0 L 587.0 365.0 L 561.0 351.0 L 545.0 346.0 L 537.0 346.0 L 526.0 349.0 L 477.0 376.0 L 469.0 378.0 L 503.0 354.0 L 515.0 348.0 L 570.0 294.0 L 579.0 282.0 L 586.0 268.0 L 588.0 260.0 L 588.0 248.0 L 586.0 242.0 L 578.0 235.0 L 568.0 237.0 L 474.0 321.0 L 434.0 370.0 L 412.0 402.0 L 396.0 430.0 L 395.0 424.0 L 401.0 394.0 L 402.0 397.0 L 404.0 397.0 L 404.0 395.0 L 401.0 394.0 L 402.0 385.0 L 411.0 352.0 L 426.0 317.0 L 426.0 308.0 L 431.0 283.0 L 431.0 269.0 L 428.0 256.0 L 424.0 248.0 L 419.0 243.0 Z M 524.0 192.0 L 514.0 191.0 L 507.0 197.0 L 434.0 313.0 L 417.0 355.0 L 406.0 399.0 L 408.0 399.0 L 425.0 372.0 L 453.0 336.0 L 476.0 312.0 L 500.0 291.0 L 523.0 252.0 L 530.0 237.0 L 534.0 220.0 L 532.0 203.0 Z"
                            fill="url(#handGrad)"
                            fillRule="evenodd"
                        />
                    </AnimatedG>

                    <AnimatedG opacity={bubbleOpacity}>
                        <Path
                            d="M 903.0 497.0 L 893.0 488.0 L 883.0 488.0 L 874.0 494.0 L 867.0 509.0 L 866.0 528.0 L 850.0 563.0 L 838.0 582.0 L 823.0 601.0 L 798.0 625.0 L 805.0 691.0 L 804.0 697.0 L 742.0 659.0 L 708.0 667.0 L 679.0 670.0 L 649.0 669.0 L 630.0 666.0 L 602.0 658.0 L 583.0 650.0 L 558.0 636.0 L 540.0 623.0 L 540.0 660.0 L 562.0 674.0 L 587.0 685.0 L 608.0 692.0 L 637.0 698.0 L 656.0 700.0 L 688.0 700.0 L 713.0 697.0 L 740.0 691.0 L 831.0 744.0 L 834.0 744.0 L 827.0 640.0 L 854.0 612.0 L 877.0 578.0 L 888.0 556.0 L 899.0 527.0 L 904.0 509.0 Z"
                            fill="url(#aquaGrad)"
                            fillRule="evenodd"
                        />
                        <Path d="M 665.0 457.0 L 665.0 463.0 L 668.0 466.0 L 672.0 467.0 L 819.0 467.0 L 822.0 466.0 L 825.0 463.0 L 825.0 457.0 L 820.0 453.0 L 669.0 453.0 Z" fill="#003D86" fillRule="evenodd" />
                        <Path d="M 660.0 507.0 L 660.0 510.0 L 664.0 515.0 L 818.0 515.0 L 822.0 512.0 L 822.0 506.0 L 818.0 502.0 L 665.0 502.0 Z" fill="#003D86" fillRule="evenodd" />
                        <Path d="M 655.0 556.0 L 655.0 559.0 L 659.0 564.0 L 746.0 564.0 L 750.0 560.0 L 750.0 555.0 L 746.0 551.0 L 660.0 551.0 Z" fill="#003D86" fillRule="evenodd" />
                    </AnimatedG>

                    <AnimatedG opacity={dotOpacity}>
                        <Circle cx="700" cy="220" r="8" fill="url(#aquaGrad)" />
                        <Circle cx="744" cy="224" r="9" fill="url(#aquaGrad)" />
                        <Circle cx="792" cy="246" r="13" fill="url(#aquaGrad)" />
                        <Circle cx="835" cy="281" r="15" fill="url(#aquaGrad)" />
                        <Circle cx="868" cy="327" r="16" fill="url(#aquaGrad)" />
                        <Circle cx="889" cy="383" r="18" fill="url(#aquaGrad)" />
                        <Circle cx="896" cy="445" r="19" fill="url(#aquaGrad)" />
                    </AnimatedG>

                    <AnimatedG opacity={wordOpacity}>
                        <Text
                            x="626"
                            y="980"
                            fill="#f6fafc"
                            fontFamily="Arial"
                            fontSize="176"
                            fontWeight="900"
                            textAnchor="middle"
                            letterSpacing="18"
                        >
                            SANKET
                        </Text>
                        <Text
                            x="626"
                            y="1060"
                            fill="#24d6be"
                            fontFamily="Arial"
                            fontSize="36"
                            fontWeight="700"
                            textAnchor="middle"
                        >
                            SIGN TO TEXT CONNECTING COMMUNITIES
                        </Text>
                    </AnimatedG>
                </Svg>
            </Animated.View>
        </View>
    );
}

const styles = StyleSheet.create({
    logoMotion: {
        height: "100%",
        width: "100%",
    },
    wrap: {
        backgroundColor: "transparent",
    },
});
