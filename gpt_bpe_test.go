package gpt_bpe

import (
	"bufio"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/wbrown/gpt_bpe/types"

	"github.com/stretchr/testify/assert"
	"github.com/ulikunitz/xz"
	"github.com/wbrown/gpt_bpe/resources"
)

var clipEncoder GPTEncoder
var gpt2Encoder GPTEncoder
var pileEncoder GPTEncoder
var nerdstashV2Encoder GPTEncoder
var llama2Encoder GPTEncoder
var llama3Encoder GPTEncoder
var mistralEncoder GPTEncoder
var corpus string
var clipCorpus string
var largeCorpus *string

// var corpus2 string
var gpt2Encoded *Tokens
var pileEncoded *Tokens
var clipEncoded *Tokens
var nerdstashEncoded *Tokens
var llama2Encoded *Tokens
var llama3Encoded *Tokens
var mistralEncoded *Tokens
var unicodeTrimTests []*Tokens

var benchmarkPrefix string
var encoders map[string]*GPTEncoder

const largeCorpusPath = "resources/wiki.train.raw.xz"

func handleRead(path string) []byte {
	if textBytes, err := os.ReadFile(path); err != nil {
		log.Fatalf("Error opening `%s`: %v", path, err)
	} else {
		return textBytes
	}
	return nil
}

func loadUnicodeTrimTests(path string) []*Tokens {
	tests := make([]*Tokens, 0)
	fileBlob := string(handleRead(path))
	fileLines := strings.Split(fileBlob, "\n")
	for idx := range fileLines {
		line := fileLines[idx]
		if len(line) == 0 {
			continue
		}
		unicodeTrimTest := make(Tokens, 0)
		if err := json.Unmarshal(
			[]byte(line),
			&unicodeTrimTest,
		); err != nil {
			log.Fatalf("Error unmarshaling `%s`: %v", path, err)
		}
		tests = append(tests, &unicodeTrimTest)
	}
	return tests
}

func Chunks(s string, chunkSize int) []string {
	if len(s) == 0 {
		return nil
	}
	if chunkSize >= len(s) {
		return []string{s}
	}
	var chunks []string = make([]string, 0, (len(s)-1)/chunkSize+1)
	currentLen := 0
	currentStart := 0
	for i := range s {
		if currentLen == chunkSize {
			chunks = append(chunks, s[currentStart:i])
			currentLen = 0
			currentStart = i
		}
		currentLen++
	}
	chunks = append(chunks, s[currentStart:])
	return chunks
}

func getStringBounds(
	i int,
	output string,
	decoded string,
) (
	left int,
	right int,
) {
	if i < 20 {
		left = 0
	} else {
		left = i - 20
	}
	if len(output) < len(decoded) {
		right = len(output)
	} else {
		right = len(decoded)
	}
	if i+20 < right {
		right = i + 20
	}
	return left, right
}

func init() {
	isBench := isRunningBenchmarkTest()

	// These will all be null
	encoders = map[string]*GPTEncoder{
		"gpt2-tokenizer":         &gpt2Encoder,
		"pile-tokenizer":         &pileEncoder,
		"clip-tokenizer":         &clipEncoder,
		"nerdstash_v2-tokenizer": &nerdstashV2Encoder,
		"llama-tokenizer":        &llama2Encoder,
		"llama3-tokenizer":       &llama3Encoder,
		"mistral-tokenizer":      &mistralEncoder,
	}

	// Load all encoders up front, as this is desirable for benchmarking
	if isBench {
		gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")
		pileEncoder = *CacheLoadEncoder("pile-tokenizer")
		clipEncoder = *CacheLoadEncoder("clip-tokenizer")
		nerdstashV2Encoder = *CacheLoadEncoder("nerdstash_v2-tokenizer")
		llama2Encoder = *CacheLoadEncoder("llama-tokenizer")
		llama3Encoder = *CacheLoadEncoder("llama3-tokenizer")
		mistralEncoder = *CacheLoadEncoder("mistral-tokenizer")
	}

	textBytes := handleRead("resources/frankenstein.txt")
	clipBytes := handleRead("resources/frankenstein_clip.txt")
	corpus = string(textBytes)
	clipCorpus = string(clipBytes)
	unicodeTrimTests = loadUnicodeTrimTests("resources/trim_tests.jsonl")
	var err error
	_, err = GetLargeCorpus()
	if err != nil {
		log.Fatalf("Error opening `%s`: %v", largeCorpusPath, err)
	}
}

// isRunningBenchmarkTest
// Check if we're running a benchmark test
// We assume a benchmark test is defined as a test
// that begins with a BENCHMARK_PREFIX. This is
// by default "Benchmark", but can be configured
// by the environment variable BENCHMARK_PREFIX
func isRunningBenchmarkTest() bool {
	prefix, ok := os.LookupEnv("BENCHMARK_PREFIX")
	if ok {
		benchmarkPrefix = prefix
	} else {
		benchmarkPrefix = "Benchmark"
	}

	for _, arg := range os.Args {
		parsedArg, err := regexp.Compile(arg)
		if err != nil {
			log.Fatalf("Failed parsing CLI args using regexp")
		}
		prefix, _ = parsedArg.LiteralPrefix()
		if strings.HasPrefix(prefix, benchmarkPrefix) {
			log.Println("Running benchmark test, so loading encoders up front...")
			return true
		}
	}
	return false
}

// CacheLoadEncoder
// Loads an encoder, but only once
func CacheLoadEncoder(vocabId string) *GPTEncoder {
	if encoders[vocabId].Encoder == nil {
		encoder, err := NewEncoder(vocabId)
		if err != nil {
			log.Fatalf("Error loading encoder `%s`: ", vocabId)
		}

		// Cache the encoder for later use
		encoders[vocabId] = encoder
		return encoder
	}
	return encoders[vocabId]
}

func GetLargeCorpus() (*string, error) {
	if largeCorpus == nil {
		var err error
		largeCorpus, err = DecompressXZ(largeCorpusPath)
		if err != nil {
			return nil, err
		}
	}
	return largeCorpus, nil
}

func TestMain(m *testing.M) {
	m.Run()
}

type TrimTest struct {
	Input     string
	Direction TrimDirection
	Limit     uint
	Expected  string
}

const sent1 = "This is test sentence 1.  This is test sentence 2.  This is test sentence 3."
const sent2 = "\nThis is test sentence 4.\nThis is test sentence 5.\nThis is test sentence 6.\n"
const hindiSentence = "व्याकरण शास्त्रीय परिभाषाएँ : डॉ. पर्णदत्त सिंह द्वारा हिंदी पीडीऍफ़ पुस्तक"
const jpSentence = "「そんな心構えで、本当に俺の『未練』を果たせるのか？　知ってのとおり、俺の『未練』は『<|rubycover|>相川渦波<|rubystart|>おまえ<|rubyend|>の成長を最後まで見届けること』だ。……言っとくが、俺は年季が入ってる上に、拗らせに拗らせた元神学者。俺の『大いなる<|rubycover|>救世主<|rubystart|>マグナ・メサイア<|rubyend|>』の『理想』は高いぞ？　少なくとも、この『血陸』を止められないようじゃ、任せ切れないな」\n<|mtsentence|><|mtsenglish|>Please check if the meat is being roasted at the right heat.<|mtsjapanese|>焼き肉の火加減を見なさい。<|mtsentenceend|>\n<|mtvocab|><|mtvjapanese|>[ぶんけんがく] 文献学<|mtvenglish|>(n) philology<|mtvocabend|>"

var TrimSentencesTests = []TrimTest{
	{sent1, TrimTop, 10,
		" This is test sentence 3."},
	{sent1, TrimTop, 20,
		" This is test sentence 2.  This is test sentence 3."},
	{sent1, TrimTop, 30,
		sent1},
	{sent2, TrimTop, 10,
		"\nThis is test sentence 6.\n"},
	{sent2, TrimTop, 18,
		"\nThis is test sentence 5.\nThis is test sentence 6.\n"},
	{sent2, TrimTop, 30,
		sent2},
	{sent1, TrimBottom, 10,
		"This is test sentence 1."},
	{sent1, TrimBottom, 20,
		"This is test sentence 1.  This is test sentence 2."},
	{sent1, TrimBottom, 30,
		sent1},
	{sent2, TrimBottom, 10,
		"\nThis is test sentence 4.\n"},
	{sent2, TrimBottom, 18,
		"\nThis is test sentence 4.\nThis is test sentence 5.\n"},
	{sent2, TrimBottom, 30,
		sent2},
}

func TestHFResolution(t *testing.T) {
	_, err := NewEncoder("EleutherAI/gpt-j-6B")
	if err != nil {
		t.Error(err)
	}
	_, err = NewEncoder("nonexist/nonexist")
	if err == nil {
		t.Error(errors.New("failed to return error on non-existent model"))
	}
}

func TestHFTokenzier(t *testing.T) {
	enc, err := NewEncoder("EleutherAI/gpt-j-6B")
	if err != nil {
		t.Error(err)
	}
	sent := "The fox jumped over the hare."
	hfTokens := enc.Encode(&sent)
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")
	gptTokens := gpt2Encoder.Encode(&sent)
	assert.Equal(t, hfTokens, gptTokens)
}

func TestFairSeqTokenizer(t *testing.T) {
	enc, err := NewEncoder("KoboldAI/fairseq-dense-2.7B")
	if err != nil {
		t.Error(err)
		return
	}
	sent := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	tokens := Tokens{464, 21831, 11687, 625, 262, 387, 260, 25970, 82, 29,
		464, 28699, 318, 5443, 621, 262, 387, 260, 13}
	fsTokens := enc.Encode(&sent)
	assert.Equal(t, *fsTokens, tokens)
}

var TrimNewLinesTests = append(
	TrimSentencesTests[3:5], TrimSentencesTests[9:11]...,
)

func TestGPTEncoder_TrimIncompleteSentence(t *testing.T) {
	testStr := "This is a test. He says, \"This is an unterminated quote. She says, this is actually terminated.\" This is awesome! This is incomplete "
	expected := "This is a test. He says, \"This is an unterminated quote. She says, this is actually terminated.\" This is awesome!"
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")
	trimmed, _ := gpt2Encoder.TrimIncompleteSentence(gpt2Encoder.Encode(&testStr))
	output := gpt2Encoder.Decode(trimmed)
	if expected != output {
		t.Error("output != expected; output := ", expected)
	}
}

func TestGPTEncoder_TrimTokens(t *testing.T) {
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")
	for testIdx := range unicodeTrimTests {
		assert.NotEqual(
			t, len(
				*gpt2Encoder.TrimTokens(
					unicodeTrimTests[testIdx],
				),
			),
			len(*unicodeTrimTests[testIdx]),
		)
	}
}

func TestGPTEncoder_TrimNewlines(t *testing.T) {
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")
	for testIdx := range TrimNewLinesTests {
		test := TrimNewLinesTests[testIdx]
		res, err := gpt2Encoder.TrimNewlines(
			gpt2Encoder.Encode(&test.Input),
			test.Direction, test.Limit,
		)
		if err != nil {
			t.Error("TrimNewlines: error:", err)
		}
		decodeRes := gpt2Encoder.Decode(res)
		if decodeRes != test.Expected {
			t.Error(
				"TrimNewlines: expected '" + test.Expected + "' got '" +
					decodeRes + "'",
			)
		}
	}
}

func TestGPTEncoder_TrimSentences(t *testing.T) {
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")
	for testIdx := range TrimSentencesTests {
		test := TrimSentencesTests[testIdx]
		res, err := gpt2Encoder.TrimSentences(
			gpt2Encoder.Encode(&test.Input),
			test.Direction, test.Limit,
		)
		if err != nil {
			t.Error("TrimSentences: error:", err)
		}
		decodeRes := gpt2Encoder.Decode(res)
		if decodeRes != test.Expected {
			t.Error(
				"TrimSentences: expected '" + test.Expected + "' got '" +
					decodeRes + "'",
			)
		}
	}
}

type SplitTest struct {
	Input    string
	Expected []string
}

var SplitTests = []SplitTest{
	{"we'll go jump in a lake.",
		[]string{"we", "'ll", " go", " jump", " in", " a", " lake",
			"."}},
	{"multiple  encoded spaces.",
		[]string{"multiple", "  ", "encoded", " spaces", "."}},
	{"Capitalized Words Are Cool",
		[]string{"Capitalized", " Words", " Are", " Cool"}},
	{"we'LL test irregular cApitalizatioN.",
		[]string{"we", "'", "LL", " test", " irregular",
			" cApitalizatioN", "."}},
	{"multilines\nare awesome",
		[]string{"multilines", "\n", "are", " awesome"}},
	{"\nstarting with multilines\nis awesome",
		[]string{"\n", "starting", " with", " multilines",
			"\n", "is", " awesome"}},
	{"we'll go jump<|endoftext|> in a lake.",
		[]string{"we", "'ll", " go", " jump", "<|endoftext|>",
			" in", " a", " lake", "."}},
	{"we'll go jump<|end\noftext|> in a lake.",
		[]string{"we", "'ll", " go", " jump", "<|", "end", "\n",
			"oftext", "|>", " in", " a", " lake", "."}},
}

func TestGPTEncoder_Split(t *testing.T) {
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")

	for testIdx := range SplitTests {
		test := SplitTests[testIdx]
		assert.Equal(t, test.Expected, *(gpt2Encoder.SplitWords(&test.Input)))
	}
}

func DecompressXZ(path string) (*string, error) {
	corpusHandle, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer corpusHandle.Close()
	decompressorHandle, err := xz.NewReader(corpusHandle)
	if err != nil {
		return nil, err
	}
	decompressed, err := ioutil.ReadAll(decompressorHandle)
	if err != nil {
		return nil, err
	}
	decompressedStr := string(decompressed)
	return &decompressedStr, nil
}

func OpenXZStream(path string) (*xz.Reader, *os.File, error) {
	corpusHandle, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	decompressorHandle, err := xz.NewReader(corpusHandle)
	if err != nil {
		return nil, nil, err
	}
	return decompressorHandle, corpusHandle, nil
}

type EncoderTest struct {
	Input             string
	GPT2Expected      Tokens
	PileExpected      Tokens
	CLIPExpected      Tokens
	NerdstashExpected Tokens
}

var GPTEncoderTests = []EncoderTest{
	{"… …",
		Tokens{1399, 3926},
		Tokens{2866, 8139},
		Tokens{49406, 959, 959, 49407},
		Tokens{49289, 5512}},
	{"<|endoftext|>",
		Tokens{50256},
		Tokens{0},
		Tokens{49406, 49407, 49407},
		Tokens{3}},
	{" <|endoftext|>\n<|endoftext|>foo",
		Tokens{220, 50256, 198, 50256, 21943},
		Tokens{209, 0, 187, 0, 12110},
		Tokens{49406, 49407, 49407, 23435, 49407},
		Tokens{49209, 3, 85, 3, 49225, 3292}},
	{" <|padding|>test",
		Tokens{220, 50257, 9288},
		Tokens{209, 1, 2566},
		Tokens{49406, 27, 347, 3798, 796, 91, 285, 1628, 49407},
		Tokens{3252, 49376, 42545, 49376, 49405, 10180},
	},
}

func TestGPTEncoder_Encode(t *testing.T) {
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")

	// This test is to check if the GPTEncoder is able to encode the tokens correctly
	start := time.Now()
	tokenCt := len(*gpt2Encoder.Encode(&corpus))
	duration := time.Since(start)
	t.Logf(
		"%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration,
	)
	for testIdx := range GPTEncoderTests {
		tokensPtr := *gpt2Encoder.Encode(
			&(GPTEncoderTests[testIdx].Input),
		)
		assert.Equal(t, tokensPtr, GPTEncoderTests[testIdx].GPT2Expected)
	}
}

func TestGPTEncoder_StreamingEncode(t *testing.T) {
	// This test is to check if the GPTEncoder is able to encode the tokens
	// correctly
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")
	start := time.Now()
	corpusRunes := strings.NewReader(*largeCorpus)
	// Set our profiler up
	profileHandle, _ := os.Create("streaming.prof")
	defer profileHandle.Close()
	runtime.GC()
	//pprof.StartCPUProfile(profileHandle)
	nextTokens := gpt2Encoder.StreamingEncode(corpusRunes)
	tokenCt := 0
	for {
		tokens := nextTokens(16384)
		if tokens == nil {
			break
		}
		tokenCt += len(*tokens)
	}
	duration := time.Since(start)
	//pprof.StopCPUProfile()
	t.Logf(
		"streaming encode: %d tokens/sec",
		int64(float64(tokenCt)/duration.Seconds()),
	)
}

func TestCLIPEncoder_Encode(t *testing.T) {
	clipEncoder = *CacheLoadEncoder("clip-tokenizer")

	// This test is to check if the CLIPEncoder is able to encode the tokens correctly
	start := time.Now()
	tokenCt := len(*clipEncoder.Encode(&corpus))
	duration := time.Since(start)
	t.Logf(
		"%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration,
	)
	for testIdx := range GPTEncoderTests {
		testStr := GPTEncoderTests[testIdx].Input
		tokensPtr := *clipEncoder.Encode(&testStr)
		assert.Equal(t, GPTEncoderTests[testIdx].CLIPExpected, tokensPtr)
	}
}

func TestPileEncoder_Encode(t *testing.T) {
	pileEncoder = *CacheLoadEncoder("pile-tokenizer")

	// This test is to check if the PileEncoder is able to encode the tokens correctly
	start := time.Now()
	tokenCt := len(*pileEncoder.Encode(&corpus))
	duration := time.Since(start)
	t.Logf(
		"%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration,
	)
	for testIdx := range GPTEncoderTests {
		tokensPtr := *pileEncoder.Encode(
			&(GPTEncoderTests[testIdx].Input),
		)
		assert.Equal(t, GPTEncoderTests[testIdx].PileExpected, tokensPtr)
	}
}

func TestNerdstashEncoder_Encode(t *testing.T) {
	// This test is to check if the NerdstashEncoder is able to encode the tokens correctly
	start := time.Now()
	nerdstashV2Encoder = *CacheLoadEncoder("nerdstash_v2-tokenizer")

	tokenCt := len(*nerdstashV2Encoder.Encode(&corpus))
	duration := time.Since(start)
	t.Logf(
		"%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration,
	)
	for testIdx := range GPTEncoderTests {
		tokensPtr := *nerdstashV2Encoder.Encode(
			&(GPTEncoderTests[testIdx].Input),
		)
		assert.Equal(t, GPTEncoderTests[testIdx].NerdstashExpected, tokensPtr)
	}
}

func TestNerdstashEncoder_EncodeSpaces(t *testing.T) {
	nerdstashV2Encoder = *CacheLoadEncoder("nerdstash_v2-tokenizer")

	// This test is to check if the NerdstashEncoder is able to encode spaces correctly
	testString := "        12 => '',\n"
	expected := Tokens{16, 124, 125, 10631, 1695, 49231, 85}
	encoded := nerdstashV2Encoder.Encode(&testString)
	assert.Equal(t, expected, *encoded)
}

func TestNerdstashEncoder_Encode2(t *testing.T) {
	nerdstashV2Encoder = *CacheLoadEncoder("nerdstash_v2-tokenizer")

	// read the jsonl test file in
	testFile, err := os.Open("resources/subset.jsonl")
	if err != nil {
		t.Error(err)
	}
	defer testFile.Close()
	scanner := bufio.NewScanner(testFile)
	scanner.Split(bufio.ScanLines)
	type testLineStruct struct {
		Text    *string `json:"text"`
		Hex     *string `json:"hex"`
		Encoded Tokens  `json:"encoded"`
	}

	passCt := 0
	failCt := 0

	for scanner.Scan() {
		jsonLine := scanner.Text()
		testLine := testLineStruct{}
		err := json.Unmarshal([]byte(jsonLine), &testLine)
		if err != nil {
			t.Error(err)
		}
		expected := testLine.Encoded
		var inputStr string
		if testLine.Hex != nil {
			inputBytes, hexErr := hex.DecodeString(*testLine.Hex)
			if hexErr != nil {
				t.Error(hexErr)
			}
			inputStr = string(inputBytes)
		} else {
			inputStr = *testLine.Text
		}
		// encode the string
		encoded := nerdstashV2Encoder.Encode(&inputStr)
		// check that the encoded string is the same as the expected
		if !assert.Equal(t, expected, *encoded) {
			t.Logf("failure on input: `%v`", inputStr)
			expectedRepr := []string{}
			for _, token := range expected {
				expectedRepr = append(
					expectedRepr,
					string(nerdstashV2Encoder.Decoder[token]),
				)
			}
			actualRepr := []string{}
			for _, token := range *encoded {
				actualRepr = append(
					actualRepr,
					string(nerdstashV2Encoder.Decoder[token]),
				)
			}
			t.Logf("expected: |%s", strings.Join(expectedRepr, "|"))
			t.Logf("actual:   |%s", strings.Join(actualRepr, "|"))
			failCt += 1
		} else {
			passCt += 1
		}
	}
	t.Logf("pass: %v, fail: %v", passCt, failCt)
}

func TestNerdstashEncoder_Decode(t *testing.T) {
	nerdstashV2Encoder = *CacheLoadEncoder("nerdstash_v2-tokenizer")

	// This test is to check if the NerdstashEncoder is able to decode the tokens correctly
	for testIdx := range GPTEncoderTests {
		decodedStr := nerdstashV2Encoder.Decode(
			&(GPTEncoderTests[testIdx].NerdstashExpected),
		)
		assert.Equal(t, GPTEncoderTests[testIdx].Input, decodedStr)
	}
}

func TestGPTEncoder_Decode2(t *testing.T) {
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")

	// This test is to check if the GPTEncoder is able to decode the tokens correctly from a base64 encoded string
	gpt2EncodedCorpus := "NrGIEOQBRzFfAQEBCAE5GeADPCFGAQhdBgFhBkcHXwEBATM5HgGilUYBpAdDEaUheR8iAQEBmgSnbyQpRgHIjaYBiSQYLfoHYwHogg0A0AHsGFUmpgEGAcd0qApjAzwa7hscAeHAYwEGAbYRB3UiAax0PQPjAgoXpgEGAZgE6G2gAWMExy5GAb5szQdGAXUBAR2gAVQBRgG8CdYBYbCgAe4QAxg/NA0AdyoiAZMGOXL8AWlmAQGgFXknNlIGAdADLiciAT4B6lk="
	decodedCorpus := "frying whatever they touched with a sizzled smell that fills the air along with a shower of sparks that land harmlessly elsewhere and a few stray drops that drip from fingers burned black as charcoal.The shock waves from the blasts cause many nearby trees to topple as the earth shakes and trembles underfoot from the power unleashed by each blast that destroys anything that was struck by it that wasn't shielded by heavy metal plates."
	if binTokens, err := base64.StdEncoding.DecodeString(gpt2EncodedCorpus); err != nil {
		log.Println("ERROR:", err)
	} else {
		tokens := types.TokensFromBin(&binTokens)
		tokens, err = gpt2Encoder.TrimIncompleteSentence(tokens)
		if err != nil {
			t.Error(err)
		}
		assert.Equal(t, gpt2Encoder.Decode(tokens), decodedCorpus)
	}
}

func TestGPTEncoder_Decode(t *testing.T) {
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")

	// This test is to check if the GPTEncoder is able to decode the tokens correctly
	if gpt2Encoded == nil {
		corpEncoded := gpt2Encoder.Encode(&corpus)
		gpt2Encoded = corpEncoded
	}
	start := time.Now()
	decoded := gpt2Encoder.Decode(gpt2Encoded)
	duration := time.Since(start)
	tokenNumBytes := len(decoded)
	t.Logf(
		"%v tokens into %v bytes over %v\n",
		len(*gpt2Encoded), tokenNumBytes, duration,
	)
	assert.Equal(t, corpus, decoded)
}

// BUG: CLIP TOKENIZER has a bug that causes 'the to be split into
// "'t<w>he<w>" instead of "'<w>the<w>".  This causes the
// clipCorpus to be different from the corpus.  This is a bug in
// the CLIP tokenizer from huggingface that was used to generate
// the clipCorpus. The decoded corpus is correct in this test.
// We stop the test right before the bug.
func TestCLIPEncoder_Decode(t *testing.T) {
	clipEncoder = *CacheLoadEncoder("clip-tokenizer")

	if clipEncoded == nil {
		corpEncoded := clipEncoder.Encode(&corpus)
		clipEncoded = corpEncoded
	}
	start := time.Now()
	decoded := clipEncoder.Decode(clipEncoded)
	duration := time.Since(start)
	tokenNumBytes := len(decoded)
	idxToStop := 229550
	t.Logf(
		"%v tokens into %v bytes over %v\n", len(*clipEncoded), tokenNumBytes,
		duration,
	)
	for idx := range clipCorpus {
		if idx > idxToStop {
			break
		}

		if clipCorpus[idx] != decoded[idx] {
			t.Errorf(
				"idx: %d, clipCorpus: %v, decoded: %v\n", idx,
				clipCorpus[idx], decoded[idx],
			)
			break
		}
	}
	// assert.Equal(t, clipCorpus, decoded)
}

func TestPileEncoder_Decode(t *testing.T) {
	pileEncoder = *CacheLoadEncoder("pile-tokenizer")

	// This test is to check if the PileEncoder is able to decode the tokens correctly
	if pileEncoded == nil {
		corpEncoded := pileEncoder.Encode(&corpus)
		pileEncoded = corpEncoded
	}
	start := time.Now()
	decoded := pileEncoder.Decode(pileEncoded)
	duration := time.Since(start)
	tokenNumBytes := len(decoded)
	t.Logf(
		"%v tokens into %v bytes over %v\n",
		len(*pileEncoded), tokenNumBytes, duration,
	)
	range_data := corpus
	if len(corpus) > len(decoded) {
		range_data = decoded
	}
	if len(corpus) != len(decoded) {
		t.Errorf(fmt.Sprintf("%v != %v", len(corpus), len(decoded)))
	}
	for idx := range range_data {
		if corpus[idx] != decoded[idx] {
			t.Errorf(
				"%v != %v", clipCorpus[idx-20:idx+20],
				decoded[idx-20:idx+20],
			)
			return
		}
	}
}

func TestGPTEncoder_TokensReady(t *testing.T) {
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")

	// This test is to check if the TokensReady function is able to determine if the tokens are ready for context
	multiTokenAsterism := "⁂"
	tokens := gpt2Encoder.Encode(&multiTokenAsterism)
	fmt.Printf("Tokens: %v, len: %v\n", tokens, len(*tokens))
	var idx int
	for idx = range *tokens {
		tokenSlice := (*tokens)[0 : idx+1]
		fmt.Printf("TokenSlice: %v, len: %v\n", tokenSlice, len(tokenSlice))
		if gpt2Encoder.TokensReady(&tokenSlice) {
			break
		}
	}
	if idx < len(*tokens)-1 {
		t.Errorf(
			"Expected TokensReady on idx: %d for `%s`", idx,
			multiTokenAsterism,
		)
	}
}

func TestGPTEncoder_TokensReadyContext(t *testing.T) {
	pileEncoder = *CacheLoadEncoder("pile-tokenizer")

	// This test is to check if the TokensReady function is able to determine if the tokens are ready for context
	var tokens Tokens
	badContext, err := os.ReadFile("resources/badcontext.json")
	if err != nil {
		t.Errorf("Could not read badcontext.json: %v", err)
	}
	unmarshalErr := json.Unmarshal(badContext, &tokens)
	if unmarshalErr != nil {
		t.Errorf("Could not unmarshal badcontext.json: %v", unmarshalErr)
	}
	if !pileEncoder.TokensReady(&tokens) {
		t.Errorf("Expected TokensReady to be true for badcontext.json")
	}
}

func TestUnitrimFunctionality(t *testing.T) {
	// This test is to check if the makeUnitrimArr function is able to generate the unitrim array correctly
	for _, tokenizer := range []string{"clip-tokenizer", "gpt2-tokenizer", "pile-tokenizer"} {
		encoderFile := fmt.Sprintf(
			"resources/data/%s/encoder.json", tokenizer,
		)
		unitrimFile := fmt.Sprintf(
			"resources/data/%s/unitrim.json", tokenizer,
		)

		// make sure the files exist
		if _, err := os.Stat(encoderFile); os.IsNotExist(err) {
			t.Errorf("Could not find file %s\n", encoderFile)
		}
		if _, err := os.Stat(unitrimFile); os.IsNotExist(err) {
			t.Errorf("Could not find file %s\n", unitrimFile)
		}

		// read in the Encoder and unitrim files
		encoderBytes, err := os.ReadFile(encoderFile)
		if err != nil {
			t.Errorf("Could not read Encoder file: %v\n", err)
		}
		// unmarshal the Encoder file
		var encoder map[string]Token
		err = json.Unmarshal(encoderBytes, &encoder)
		if err != nil {
			t.Errorf("Could not unmarshal Encoder file: %v\n", err)
		}

		// read in the unitrim file
		unitrimBytes, err := os.ReadFile(unitrimFile)
		if err != nil {
			t.Errorf("Could not read unitrim file: %v\n", err)
		}
		// unmarshal the unitrim file
		var unitrim []int
		err = json.Unmarshal(unitrimBytes, &unitrim)
		if err != nil {
			t.Errorf("Could not unmarshal unitrim file: %v\n", err)
		}

		// get generated array for unitrim with the makeUnitrimArr function
		generatedArray := makeUnitrimArr(encoder)

		// check that the generated array is the same as the unitrim array
		fmt.Printf(
			"Generated array length: %d, unitrim array length: %d\n",
			len(generatedArray), len(unitrim),
		)
		if len(generatedArray) != len(unitrim) {
			t.Errorf("Generated array and unitrim array are not the same length\n")
		}

		for i := range generatedArray {
			if generatedArray[i] != unitrim[i] {
				fmt.Printf(
					"Generated array: %v and unitrim array: %v at index %d are not the same\n",
					generatedArray[i], unitrim[i], i,
				)
				fmt.Printf(
					"mismatched unicode is: %c\n", rune(generatedArray[i]),
				)
				t.Errorf("Generated array and unitrim array are not the same\n")
			}
		}

		fmt.Printf("Length and contents of generated array and unitrim array are the same\n")
	}
}

func TestLlamaEncoder_Encode(t *testing.T) {
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")

	// This test is to check if the encoder is able to encode a basic string
	start := time.Now()
	tokenCt := len(*gpt2Encoder.Encode(&corpus))
	duration := time.Since(start)
	t.Logf(
		"%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration,
	)
	for testIdx := range GPTEncoderTests {
		tokensPtr := *gpt2Encoder.Encode(
			&(GPTEncoderTests[testIdx].Input),
		)
		assert.Equal(t, tokensPtr, GPTEncoderTests[testIdx].GPT2Expected)
	}
}

func TestLlamaTwoEncoder_Encode(t *testing.T) {
	llama2Encoder = *CacheLoadEncoder("llama-tokenizer")

	// This test is to check if the encoder is able to encode a basic string
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	llamaTokens := llama2Encoder.Encode(&testString)
	assert.Equal(
		t, llamaTokens,
		&Tokens{1576, 1701, 29916, 12500, 287, 975, 278, 447, 276, 29889, 13, 1576, 260, 4227, 280, 338, 8473, 1135, 278, 447, 276, 29889},
	)
}

func TestLlamaTwoTokenizerDecode(t *testing.T) {
	llama2Encoder = *CacheLoadEncoder("llama-tokenizer")

	// This test is to check if the decoder is able to decode the tokens correctly
	outputString := "<s>The fox jumped over the hare.\nThe turtle is faster than the hare."
	llamaTokens := Tokens{1, 1576, 1701, 29916, 12500, 287, 975, 278, 447, 276, 29889, 13, 1576, 260, 4227, 280, 338, 8473, 1135, 278, 447, 276, 29889}
	output := llama2Encoder.Decode(&llamaTokens)
	assert.Equal(t, outputString, output)
}

func TestLlamaTwoEncodeDecode(t *testing.T) {
	llama2Encoder = *CacheLoadEncoder("llama-tokenizer")

	// This test is to check if the encoder is able to encode and decode a basic string
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	outputString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	llamaTokens := llama2Encoder.Encode(&testString)
	output := llama2Encoder.Decode(llamaTokens)
	assert.Equal(t, outputString, output)
}

// This is Mistral tokenizer V1, associated with 7b instruct https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
func TestMistralEncoder_Encode(t *testing.T) {
	mistralEncoder = *CacheLoadEncoder("mistral-tokenizer")

	// This test is to check if the encoder is able to encode a basic string
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	mistralTokens := mistralEncoder.Encode(&testString)
	assert.Equal(
		t, mistralTokens,
		&Tokens{1, 415, 285, 1142, 14949, 754, 272, 295, 492, 28723, 13, 1014, 261, 3525, 291, 349, 9556, 821, 272, 295, 492, 28723},
	)
}

func TestMistralTokenizerDecode(t *testing.T) {
	mistralEncoder = *CacheLoadEncoder("mistral-tokenizer")

	// This test is to check if the decoder is able to decode the tokens correctly
	outputString := "<s> The fox jumped over the hare.\nThe turtle is faster than the hare."
	mistralTokens := Tokens{1, 415, 285, 1142, 14949, 754, 272, 295, 492, 28723, 13, 1014, 261, 3525, 291, 349, 9556, 821, 272, 295, 492, 28723}
	output := mistralEncoder.Decode(&mistralTokens)
	assert.Equal(t, outputString, output)
}

func TestMistralEncodeDecode(t *testing.T) {
	mistralEncoder = *CacheLoadEncoder("mistral-tokenizer")

	// This test is to check if the encoder is able to encode and decode a basic string
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	outputString := "<s> The fox jumped over the hare.\nThe turtle is faster than the hare."
	mistralTokens := mistralEncoder.Encode(&testString)
	output := mistralEncoder.Decode(mistralTokens)
	assert.Equal(t, outputString, output)
}

func TestMistralEncodeDecodeFrankenstein(t *testing.T) {
	mistralEncoder = *CacheLoadEncoder("mistral-tokenizer")

	// This test is to check if the encoder is able to encode and decode the Frankenstein corpus
	frankensteinCorpus := "resources/frankenstein.txt"
	frankensteinText, err := os.ReadFile(frankensteinCorpus)
	if err != nil {
		t.Errorf("Error reading Frankenstein corpus: %v", err)
	}
	frankensteinString := string(frankensteinText)
	mistralTokens := mistralEncoder.Encode(&frankensteinString)
	frankensteinString = "<s>" + frankensteinString
	output := mistralEncoder.Decode(mistralTokens)
	for i := 0; i < len(output); i++ {
		if output[i] != frankensteinString[i] {
			t.Errorf(
				"Mismatch at around index %d Expected: %v, Actual: %v", i,
				string(frankensteinString[i]), string(output[i]),
			)
			break
		}
	}
}

func TestMistralEncodeDecode_Emojis(t *testing.T) {
	mistralEncoder = *CacheLoadEncoder("mistral-tokenizer")

	// This test is to check if the encoder is able to encode and decode emojis
	// Requires the ability to properly handle byte tokens in the encoder
	testString := "expensive 😦 padding ⁂ padding"
	tokens := mistralEncoder.Encode(&testString)
	output := mistralEncoder.Decode(tokens)
	testString = "<s>" + testString
	assert.Equal(t, testString, output)
}

func TestMistralEncodeDecode_LargeCorpus(t *testing.T) {
	mistralEncoder = *CacheLoadEncoder("mistral-tokenizer")

	// This test is to check if the encoder is able to encode and decode a large corpus
	referenceFile := "resources/test_references/753.txt"
	referenceBin := "resources/test_references/753_mistralv1.bin"
	referenceText, err := os.ReadFile(referenceFile)
	if err != nil {
		t.Errorf("Error reading reference file: %v", err)
	}
	referenceString := string(referenceText)
	// Need to decode the reference bin file
	referenceBinData, err := os.ReadFile(referenceBin)
	if err != nil {
		t.Errorf("Error reading reference bin file: %v", err)
	}
	referenceTokens := types.TokensFromBin32(&referenceBinData)
	// Encode the reference string
	mistralTokens := mistralEncoder.Encode(&referenceString)
	for i := 0; i < len(*mistralTokens); i++ {
		if (*mistralTokens)[i] != (*referenceTokens)[i] {
			t.Errorf(
				"Mismatch at around index %d Expected: %v, Actual: %v", i,
				(*referenceTokens)[i], (*mistralTokens)[i],
			)
		}
	}
	assert.Equal(t, mistralTokens, referenceTokens)
	// Decode the tokens to check if the decoded string is the same as the reference string
	output := mistralEncoder.Decode(mistralTokens)
	referenceString = "<s>" + referenceString
	for i := 0; i < len(output); i++ {
		if output[i] != referenceString[i] {
			fmt.Printf("Mismatch at around index %d\n", i)
			fmt.Printf("Expected: %s\n", referenceString[i-20:i+20])
			fmt.Printf("Actual: %s\n", output[i-20:i+20])
			t.Errorf(
				"Mismatch at around index %d Expected: %s, Actual: %s", i,
				string(referenceString[i]), string(output[i]),
			)
			break
		}
	}
	assert.Equal(t, referenceString, output)
}

func TestLlama3Encoder_Encode(t *testing.T) {
	llama3Encoder = *CacheLoadEncoder("llama3-tokenizer")

	// This test is to check if the encoder is able to encode a basic string
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	llamaTokens := llama3Encoder.Encode(&testString)
	fmt.Printf("Llama3 tokens: %v\n", llamaTokens)
	assert.Equal(
		t, llamaTokens,
		&Tokens{128000, 791, 39935, 27096, 927, 279, 96018, 627, 791, 37189, 374, 10819, 1109, 279, 96018, 13, 128001},
	)
}

func TestLlama3TokenizerDecode(t *testing.T) {
	llama3Encoder = *CacheLoadEncoder("llama3-tokenizer")

	// This test is to check if the decoder is able to decode the tokens correctly
	outputString := "<|begin_of_text|>The fox jumped over the hare.\nThe turtle is faster than the hare.<|end_of_text|>"
	llamaTokens := Tokens{128000, 791, 39935, 27096, 927, 279, 96018, 627, 791, 37189, 374, 10819, 1109, 279, 96018, 13, 128001}
	output := llama3Encoder.Decode(&llamaTokens)
	assert.Equal(t, outputString, output)
}

func TestLlama3EncodeDecode(t *testing.T) {
	llama3Encoder = *CacheLoadEncoder("llama3-tokenizer")

	// This test is to check if the encoder is able to encode and decode a basic string
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."
	outputString := "<|begin_of_text|>The fox jumped over the hare.\nThe turtle is faster than the hare.<|end_of_text|>"
	llamaTokens := llama3Encoder.Encode(&testString)
	output := llama3Encoder.Decode(llamaTokens)
	assert.Equal(t, outputString, output)
}

func TestLlama3EncodeDecode_Merges(t *testing.T) {
	llama3Encoder = *CacheLoadEncoder("llama3-tokenizer")

	// This test is to check if the encoder is able to merge the tokens correctly
	// If it fails, the merge function in the streaming_encode does not check for invalid merge pairs correctly
	testString := "Ah! Cornelius Agrippa! My dear Victor, d"
	outputString := "<|begin_of_text|>Ah! Cornelius Agrippa! My dear Victor, d<|end_of_text|>"
	llamaTokens := llama3Encoder.Encode(&testString)
	fmt.Printf("Llama3 tokens: %v\n", llamaTokens)
	output := llama3Encoder.Decode(llamaTokens)
	assert.Equal(t, outputString, output)
}
func TestLlama3Merge(t *testing.T) {
	llama3Encoder = *CacheLoadEncoder("llama3-tokenizer")

	// This test is to check if the encoder is able to merge the tokens correctly
	// If it fails, the merge function in the streaming_encode does not check for invalid merge pairs correctly
	//testString := "Description\ndescription\n Description\n description"
	testString := "1234"
	llamaTokens := llama3Encoder.Encode(&testString)
	decodedTokens := make([]string, len(*llamaTokens))
	for i := 0; i < len(*llamaTokens); i++ {
		decodedTokens[i] = string(llama3Encoder.Decoder[(*llamaTokens)[i]])
	}

	if len(*llamaTokens) != 4 {
		t.Errorf("Expected 4 tokens, got %d", len(*llamaTokens))
	}

	if decodedTokens[1] != "123" && decodedTokens[2] != "4" {
		t.Errorf(
			"Expected 123|4, got %s|%s",
			decodedTokens[1], decodedTokens[2],
		)
	}
}

func TestLlama3EncodeDecode_LargeCorpus(t *testing.T) {
	llama3Encoder = *CacheLoadEncoder("llama3-tokenizer")

	// This test is to check if the encoder is able to encode and decode a large corpus
	referenceFile := "resources/test_references/753.txt"
	referenceBin := "resources/test_references/753_llama3.bin"
	referenceText, err := os.ReadFile(referenceFile)
	if err != nil {
		t.Errorf("Error reading reference file: %v", err)
	}
	referenceString := string(referenceText)
	// Need to decode the reference bin file
	referenceBinData, err := os.ReadFile(referenceBin)
	if err != nil {
		t.Errorf("Error reading reference bin file: %v", err)
	}
	referenceTokens := types.TokensFromBin32(&referenceBinData)
	// Encode the reference string
	llamaTokens := llama3Encoder.Encode(&referenceString)
	for i := 0; i < len(*llamaTokens); i++ {
		if (*llamaTokens)[i] != (*referenceTokens)[i] {
			t.Errorf(
				"Mismatch at around index %d Expected: %v, Actual: %v", i,
				(*referenceTokens)[i], (*llamaTokens)[i],
			)
		}
	}
	// Check that the encoded tokens are the same as the reference tokens
	assert.Equal(t, llamaTokens, referenceTokens)
	// Decode the tokens
	output := llama3Encoder.Decode(llamaTokens)
	refDecoded := llama3Encoder.Decode(referenceTokens)
	// Check that the decoded string is the same as the reference string
	for i := 0; i < len(output); i++ {
		if output[i] != refDecoded[i] {
			left, right := getStringBounds(i, output, refDecoded)
			fmt.Printf("Mismatch at around index %d\n", i)
			fmt.Printf("Expected: %s\n", refDecoded[left:right])
			fmt.Printf("Actual: %s\n", output[left:right])
			break
		}
	}
}

func TestLlama3EncodeDecodeFrankenstein(t *testing.T) {
	llama3Encoder = *CacheLoadEncoder("llama3-tokenizer")

	// This test is to check if the encoder is able to encode and decode the Frankenstein corpus
	frankensteinCorpus := "resources/frankenstein.txt"
	frankensteinText, err := os.ReadFile(frankensteinCorpus)
	if err != nil {
		t.Errorf("Error reading Frankenstein corpus: %v", err)
	}
	frankensteinString := string(frankensteinText)
	llamaTokens := llama3Encoder.Encode(&frankensteinString)
	output := llama3Encoder.Decode(llamaTokens)
	frankensteinString = "<|begin_of_text|>" + frankensteinString + "<|end_of_text|>"
	for i := 0; i < len(output); i++ {
		if output[i] != frankensteinString[i] {
			left, right := getStringBounds(i, output, frankensteinString)
			fmt.Printf("Mismatch at around index %d\n", i)
			fmt.Printf("Expected: %s\n", frankensteinString[left:right])
			fmt.Printf("Actual: %s\n", output[left:right])
			break
		}
	}
	assert.Equal(t, frankensteinString, output)
}

func TestReadTokenizerConfig(t *testing.T) {
	// This test is to check if the encoder is able to read the tokenizer_config.json file
	// json with eos, bos, pad as strings
	jsonStr := `{"eos_token": "TC", "bos_token": "TD", "pad_token": "TE"}` //cooresponds to 6669, 10989, 5428 in pythia vocab

	//download filler model
	modelId := "EleutherAI/pythia-70m"
	destPath := "./TestReadTokenizerConfig"
	destPathPTR := &destPath
	defer os.RemoveAll(destPath)
	rsrcType, hfApiToken := resources.RESOURCETYPE_TRANSFORMERS, os.Getenv("HF_API_TOKEN")
	os.MkdirAll(destPath, 0755)
	_, rsrcErr := resources.ResolveResources(
		modelId, destPathPTR,
		resources.RESOURCE_MODEL, rsrcType, hfApiToken,
	)
	if rsrcErr != nil {
		t.Errorf("Error downloading model resources: %s", rsrcErr)
	}

	// replace tokenizer_config.json with jsonStr
	tokenizerConfigPath := destPath + "/tokenizer_config.json"
	err := os.WriteFile(tokenizerConfigPath, []byte(jsonStr), 0644)
	if err != nil {
		t.Errorf("Error writing to tokenizer_config.json: %v", err)
	}

	// read tokenizer config by encoding a string
	encoder, err := NewEncoder(destPath)
	if err != nil {
		t.Errorf("Error creating encoder: %v", err)
	}

	// check that the tokens are correct
	assert.Equal(t, encoder.EosToken, Token(6669))
	assert.Equal(t, encoder.BosToken, Token(10989))
	assert.Equal(t, encoder.PadToken, Token(5428))

	// Finish the test, allow defered cleanup
	fmt.Println("All Exists - Looks good.")
}

func TestGPT2DefaultPadding(t *testing.T) {
	gpt2Encoder = *CacheLoadEncoder("gpt2-tokenizer")

	// GPT2 defines a padding token, we test if it properly gets this token
	// corresponds to <|padding|> in the vocab
	assert.Equal(t, gpt2Encoder.PadToken, Token(50257))
	assert.Equal(t, gpt2Encoder.Encoder["<|padding|>"], Token(50257))
}

func TestPilePadding(t *testing.T) {
	pileEncoder = *CacheLoadEncoder("pile-tokenizer")

	// Pile defines a padding token, we test if it properly gets this token
	// corresponds to <|padding|> in the vocab
	assert.Equal(t, pileEncoder.PadToken, Token(1))
	assert.Equal(t, pileEncoder.Encoder["<|padding|>"], Token(1))
}

func TestClipPadding(t *testing.T) {
	clipEncoder = *CacheLoadEncoder("clip-tokenizer")

	// CLIP defines a padding token, we test if it properly gets this token
	// corresponds to <|endoftext|> in the vocab
	assert.Equal(t, clipEncoder.PadToken, Token(49407))
	assert.Equal(t, clipEncoder.Encoder["<|endoftext|>"], Token(49407))
}

func TestNerdstashPadding(t *testing.T) {
	nerdstashV2Encoder = *CacheLoadEncoder("nerdstash_v2-tokenizer")

	// Nerdstash defines a padding token, we test if it properly gets this token
	// corresponds to <|pad|> in the vocab
	assert.Equal(t, nerdstashV2Encoder.PadToken, Token(0))
	assert.Equal(t, nerdstashV2Encoder.Encoder["<|pad|>"], Token(0))
}

func TestLlamaPadding(t *testing.T) {
	llama2Encoder = *CacheLoadEncoder("llama-tokenizer")

	// Llama doesn't define a padding token, we test if it properly defaults to
	// [PAD] as 65535
	assert.Equal(t, llama2Encoder.PadToken, Token(65535))
	assert.Equal(t, llama2Encoder.Encoder["[PAD]"], Token(65535))
}

func TestMistralPadding(t *testing.T) {
	mistralEncoder = *CacheLoadEncoder("mistral-tokenizer")

	// Mistral doesn't define a padding token, we test if it properly defaults to
	// [PAD] as 65535
	assert.Equal(t, mistralEncoder.PadToken, Token(65535))
	assert.Equal(t, mistralEncoder.Encoder["[PAD]"], Token(65535))
}

func TestLlama3Padding(t *testing.T) {
	llama3Encoder = *CacheLoadEncoder("llama3-tokenizer")

	// Llama doesn't define a padding token, we test if it properly defaults to
	// [PAD] as 4294967295 due to the uint32 max value
	assert.Equal(t, llama3Encoder.PadToken, Token(4294967295))
	assert.Equal(t, llama3Encoder.Encoder["[PAD]"], Token(4294967295))
}

func TestGPTDecoder_Decode(t *testing.T) {
	// TBD
}

func TestRankPairs(t *testing.T) {
}

func downloadModel(modelId string, destPath string) error {
	// Download the model
	destPathPTR := &destPath
	rsrcType, hfApiToken := resources.RESOURCETYPE_TRANSFORMERS, os.Getenv("HF_API_TOKEN")
	os.MkdirAll(destPath, 0755)
	_, rsrcErr := resources.ResolveResources(
		modelId, destPathPTR,
		resources.RESOURCE_MODEL, rsrcType, hfApiToken,
	)
	if rsrcErr != nil {
		return rsrcErr
	}
	return nil
}

func assertFileExists(t *testing.T, filePath string) {
	if _, err := os.Stat(filePath); err != nil && os.IsNotExist(err) {
		t.Errorf("File does not exist: %s", filePath)
	} else if err != nil {
		t.Errorf("Error checking file: %v", err)
	}

}

func TestModelDownload(t *testing.T) {
	// Download the model
	modelId := "gpt2"
	destPath := "./TestModelDownload"
	err := downloadModel(modelId, destPath)
	if err != nil {
		os.RemoveAll(destPath)
		t.Errorf("Error downloading model: %v", err)
	}
	defer os.RemoveAll(destPath)

	// Check that the model files are there
	// We want to check for the presence of the following files:
	// config.json, pytorch_model.bin,
	// tokenizer.json, vocab.json

	// Check for config.json
	configPath := destPath + "/config.json"
	assertFileExists(t, configPath)

	// Check for pytorch_model.bin
	modelPath := destPath + "/pytorch_model.bin"
	assertFileExists(t, modelPath)

	// Check for tokenizer.json
	tokenizerConfigPath := destPath + "/tokenizer.json"
	assertFileExists(t, tokenizerConfigPath)

	// Check for vocab.json
	vocabPath := destPath + "/vocab.json"
	assertFileExists(t, vocabPath)

	// Finish the test, allow defered cleanup
	fmt.Println("All Exists - Looks good.")
}

func TestPythiaRemoteDownloadTokenizer(t *testing.T) {
	// Tests the ability to download a tokenizer from a remote model
	// and use it to encode and decode strings
	modelId := "EleutherAI/pythia-70m"
	destPath := "./TestPythiaRemoteDownloadTokenizer"
	defer os.RemoveAll(destPath)
	encoderPythia, err := NewEncoder(modelId)
	if err != nil {
		t.Errorf("Error creating encoder: %v", err)
	}

	// Attempt to tokenize
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."

	// Encode the string
	encoded := encoderPythia.Encode(&testString)
	// Check that the encoded string is the same as the expected - Reference from python's transformers lib
	expected := Tokens{510, 30013, 16780, 689, 253, 419, 250, 15, 187, 510, 45993, 310, 7938, 685, 253, 419, 250, 15}
	if !assert.Equal(t, expected, *encoded) {
		t.Errorf("Expected: %v\nActual: %v", expected, *encoded)
	}
}

func TestLlama3RemoteDownloadTokenizer(t *testing.T) {
	// Tests the ability to download a tokenizer from a remote model
	// and use it to encode and decode strings
	modelId := "Groq/Llama-3-Groq-8B-Tool-Use" // Original Llama3 model is gated
	destPath := "./TestLlama3RemoteDownloadTokenizer"
	defer os.RemoveAll(destPath)
	encoderLlama3, err := NewEncoder(modelId)
	if err != nil {
		t.Errorf("Error creating encoder: %v", err)
	}

	// Attempt to tokenize
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."

	// Encode the string
	encoded := encoderLlama3.Encode(&testString)
	// Check that the encoded string is the same as the expected - 128009 is 128001 in the original Llama3 model
	expected := Tokens{128000, 791, 39935, 27096, 927, 279, 96018, 627, 791, 37189, 374, 10819, 1109, 279, 96018, 13, 128009}
	if !assert.Equal(t, expected, *encoded) {
		t.Errorf("Expected: %v\nActual: %v", expected, *encoded)
	}
}

func TestMistralRemoteDownloadTokenizer(t *testing.T) {
	// Tests the ability to download a tokenizer from a remote model
	// and use it to encode and decode strings
	modelId := "Open-Orca/Mistral-7B-OpenOrca"
	//destPath := "./TestMistralRemoteDownloadTokenizer"
	//defer os.RemoveAll(destPath)
	encoderMistral, err := NewEncoder(modelId)
	if err != nil {
		t.Errorf("Error creating encoder: %v", err)
	}

	// Attempt to tokenize
	testString := "The fox jumped over the hare.\nThe turtle is faster than the hare."

	// Encode the string
	encoded := encoderMistral.Encode(&testString)
	// Check that the encoded string is the same as the expected - Reference from python's transformers lib
	expected := Tokens{1, 1014, 285, 1142, 14949, 754, 272, 295, 492, 28723, 13, 1014, 261, 3525, 291, 349, 9556, 821, 272, 295, 492, 28723}
	if !assert.Equal(t, expected, *encoded) {
		t.Errorf("Expected: %v\nActual: %v", expected, *encoded)
	}
}

func TestModelDownloadPythia(t *testing.T) {
	// Pythia uses a slightly different file structure, where
	// the vocab.json and merges.txt files are stored in the
	// tokenizer.json file. We want to check if we are able to
	// download the model and extract the vocab.json and merges.txt
	modelId := "EleutherAI/pythia-70m"
	destPath := "./TestModelDownloadPythia"
	err := downloadModel(modelId, destPath)
	if err != nil {
		os.RemoveAll(destPath)
		t.Errorf("Error downloading model: %v", err)
	}

	// Check that the model files are there
	// We want to check for the presence of the following files:
	// config.json, pytorch_model.bin,
	// tokenizer.json, vocab.json

	// Check for additional metadata files
	metaFiles := []string{"tokenizer.json", "vocab.json", "config.json", "pytorch_model.bin"}
	for _, metaFile := range metaFiles {
		metaPath := destPath + "/" + metaFile
		assertFileExists(t, metaPath)
	}

	// Finish the test, allow defered cleanup
	fmt.Println("All Exists - Looks good.")
}

func TestModelDownloadPythiaSharded(t *testing.T) {
	// This tests the model downloader's ability
	// to download a sharded model.

	modelId := "EleutherAI/pythia-6.9b-deduped"
	destPath := "./TestModelDownloadPythiaSharded"
	err := downloadModel(modelId, destPath)
	if err != nil {
		os.RemoveAll(destPath)
		t.Errorf("Error downloading model: %v", err)
	}
	defer os.RemoveAll(destPath)

	// Check that the model files are there
	// We want to check for the presence of the following files:
	// pytorch_model-00001-of-00002.bin, pytorch_model-00002-of-00002.bin,
	// pytorch_model.bin.index.json

	// Check for pytorch_model-00001-of-00002.bin
	model1Path := destPath + "/pytorch_model-00001-of-00002.bin"
	assertFileExists(t, model1Path)

	// Check for pytorch_model-00002-of-00002.bin
	model2Path := destPath + "/pytorch_model-00002-of-00002.bin"
	assertFileExists(t, model2Path)

	// Check for pytorch_model.bin.index.json
	shardconfigPath := destPath + "/pytorch_model.bin.index.json"
	assertFileExists(t, shardconfigPath)

	// Finish the test, allow defered cleanup
	fmt.Println("All Exists - Looks good.")

}

func TestModelDownloadLlama(t *testing.T) {
	// Pythia uses a slightly different file structure, where
	// the vocab.json and merges.txt files are stored in the
	// tokenizer.json file. We want to check if we are able to
	// download the model and extract the vocab.json and merges.txt
	modelId := "georgesung/llama2_7b_chat_uncensored"
	destPath := "./TestModelDownloadLlama"
	err := downloadModel(modelId, destPath)
	if err != nil {
		os.RemoveAll(destPath)
		t.Errorf("Error downloading model: %v", err)
	}
	defer os.RemoveAll(destPath)

	// Check that the model files are there
	// We want to check for the presence of the following files:
	// config.json, pytorch_model.bin,
	// tokenizer.json, vocab.json

	// Check for pytorch_model.bin
	singleModelPattern := regexp.MustCompile(`pytorch_model\.bin$`)
	re, err := regexp.Compile(`-(\d+)-of-(\d+)\.bin$`)
	if err != nil {
		t.Errorf("Error compiling regex: %s", err)
	}

	//check all files in the directory against the pattern
	files, err := ioutil.ReadDir(destPath)
	if err != nil {
		t.Errorf("Error reading directory: %s", err)
	}
	found := false

	for _, file := range files {
		if singleModelPattern.MatchString(file.Name()) {
			found = true
			break
		}

		matches := re.FindStringSubmatch(file.Name())
		if len(matches) > 2 {
			if strings.Compare(matches[1], matches[2]) == 0 {
				found = true
				break
			}
		}
	}
	if !found {
		t.Errorf("pytorch_model.bin does not exist or was not found")
	}

	// Check for additional metadata files
	metaFiles := []string{"tokenizer.json", "vocab.json", "config.json"}
	for _, metaFile := range metaFiles {
		metaPath := destPath + "/" + metaFile
		assertFileExists(t, metaPath)
	}

	// Finish the test, allow defered cleanup
	fmt.Println("All Exists - Looks good.")
}

func TestModelDownloadMistral(t *testing.T) {
	// Download a downstream mistral model due to mistral being gated
	modelId := "Open-Orca/Mistral-7B-OpenOrca"
	destPath := "./TestModelDownloadMistral"
	err := downloadModel(modelId, destPath)
	if err != nil {
		os.RemoveAll(destPath)
		t.Errorf("Error downloading model: %v", err)
	}
	defer os.RemoveAll(destPath)

	// Check that the model files are there
	// We want to check for the presence of the following files:
	// config.json, pytorch_model.bin,
	// tokenizer.json, vocab.json

	// Check for additional metadata files
	metaFiles := []string{"tokenizer.json", "vocab.json, config.json", "pytorch_model-00001-of-00002.bin"}
	for _, metaFile := range metaFiles {
		metaPath := destPath + "/" + metaFile
		assertFileExists(t, metaPath)
	}

	// Finish the test, allow defered cleanup
	fmt.Println("All Exists - Looks good.")
}

func TestModelDownloadFairseq(t *testing.T) {
	// Koboldai's fairseq models are stored in a different format
	// it has merges and vocab but no tokenizer.json
	modelId := "KoboldAI/fairseq-dense-355M"
	destPath := "./TestModelDownloadFairseq"

	// Download the model
	err := downloadModel(modelId, destPath)
	if err != nil {
		os.RemoveAll(destPath)
		t.Errorf("Error downloading model: %v", err)
	}
	defer os.RemoveAll(destPath)

	// Check that the model files are there
	// We want to check for the presence of the following files:
	// vocab, config. merges, pytorch_model

	// Check for additional metadata files
	metaFiles := []string{"tokenizer.json", "vocab.json, config.json", "pytorch_model.bin", "merges.txt"}
	for _, metaFile := range metaFiles {
		metaPath := destPath + "/" + metaFile
		assertFileExists(t, metaPath)
	}

	// Finish the test, allow defered cleanup
	fmt.Println("All Exists - Looks good (Fairseq Download).")
}
