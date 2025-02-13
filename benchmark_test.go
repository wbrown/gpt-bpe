package gpt_bpe

import (
	"bufio"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"
	"testing"
	"time"
)

func BenchmarkGPTEncoder_WordSplitterChan(b *testing.B) {
	b.StopTimer()
	gpt2Encoder.SplitterThreads = 8
	corpusHandle := strings.NewReader(corpus)
	nextWord := gpt2Encoder.WordSplitter(
		bufio.NewReaderSize(
			corpusHandle, 8*1024*1024,
		),
	)

	start := time.Now()
	b.StartTimer()
	wordCount := 0
	for {
		word := nextWord()
		if word == nil {
			break
		}
		wordCount++
	}
	b.StopTimer()
	elapsed := time.Since(start)
	b.ReportMetric(float64(wordCount)/elapsed.Seconds(), "words/sec")
	b.ReportMetric(float64(wordCount), "words")
}

func BenchmarkGPTEncoder_WordSplitter(b *testing.B) {
	b.StopTimer()
	corpusHandle := strings.NewReader(*largeCorpus)
	gpt2Encoder.SplitterThreads = 8
	wordCount := 0
	runeReader := bufio.NewReaderSize(corpusHandle, 8*1024*1024)
	profileHandle, _ := os.Create("wordsplitter.prof")
	runtime.GC()
	pprof.StartCPUProfile(profileHandle)
	wordSplitter := gpt2Encoder.makeWordSplitter(
		runeReader.ReadRune,
		func(words []string) {
			wordCount += len(words)
		},
		func() {},
	)
	start := time.Now()
	b.StartTimer()
	wordSplitter()
	b.StopTimer()
	pprof.StopCPUProfile()
	elapsed := time.Since(start)
	numBytes := len(*largeCorpus)
	b.ReportMetric(float64(wordCount)/elapsed.Seconds(), "words/sec")
	b.ReportMetric(float64(wordCount), "words")
	b.ReportMetric(float64(numBytes)/elapsed.Seconds(), "bytes/sec")
	b.ReportMetric(float64(numBytes), "bytes")
}

func BenchmarkGPTEncoder_ToBPE(b *testing.B) {
	b.StopTimer()

	// Pre-split words
	words := *nerdstashV2Encoder.SplitWords(largeCorpus)

	// Pre-calculate tokens for each word
	tokenLengths := make([]int, len(words))
	totalTokens := 0
	for i, word := range words {
		tokens := nerdstashV2Encoder.ToBPE(word)
		tokenLengths[i] = len(tokens)
		totalTokens += tokenLengths[i]
	}
	profileHandle, _ := os.Create("tobpe.prof")

	numBytes := len(*largeCorpus)
	start := time.Now()

	b.StartTimer()
	runtime.GC()
	pprof.StartCPUProfile(profileHandle)
	for i := 0; i < b.N; i++ {
		for idx := range words {
			// Just do the ToBPE call without length calculation
			nerdstashV2Encoder.ToBPE(words[idx])
		}
	}
	pprof.StopCPUProfile()
	b.StopTimer()

	elapsed := time.Since(start)
	totalTokens *= b.N

	// Use pre-calculated values for metrics
	b.ReportMetric(float64(numBytes)/elapsed.Seconds(), "bytes/sec")
	b.ReportMetric(float64(numBytes), "bytes")
	b.ReportMetric(float64(totalTokens)/elapsed.Seconds(), "tokens/sec")
	b.ReportMetric(float64(totalTokens), "tokens")
	// Report on tokenizer LRU cache
	b.ReportMetric(float64(nerdstashV2Encoder.LruHits), "lru_hits")
	b.ReportMetric(float64(nerdstashV2Encoder.LruMisses), "lru_misses")
	b.ReportMetric(float64(nerdstashV2Encoder.LruEvictions), "lru_evictions")

}

func BenchmarkGPTEncoder_WordSplitterTokens(b *testing.B) {
	b.StopTimer()
	wordCount := 0
	tokensCount := 0
	corpusHandle := strings.NewReader(corpus)
	runeReader := bufio.NewReaderSize(corpusHandle, 8*1024*1024)

	wordSplitter := nerdstashV2Encoder.makeWordSplitter(
		runeReader.ReadRune,
		func(words []string) {
			if len(words) > 0 {
				for _, word := range words {
					tokensCount += len(nerdstashV2Encoder.ToBPE(word))
				}
			}
			wordCount++
		},
		func() {},
	)
	start := time.Now()
	b.StartTimer()
	wordSplitter()
	b.StopTimer()
	elapsed := time.Since(start)
	//numBytes := int64(len(corpusText))
	b.ReportMetric(float64(wordCount)/elapsed.Seconds(), "words/sec")
	b.ReportMetric(float64(wordCount), "words")
	b.ReportMetric(float64(tokensCount)/elapsed.Seconds(), "tokens/sec")
	b.ReportMetric(float64(tokensCount), "tokens")
}

func BenchmarkGPTEncoder_Decode(b *testing.B) {
	if gpt2Encoded == nil {
		corpEncoded := gpt2Encoder.Encode(&corpus)
		gpt2Encoded = corpEncoded
	}
	start := time.Now()
	tokenNumBytes := len(gpt2Encoder.Decode(gpt2Encoded))
	duration := time.Since(start)
	b.Logf(
		"%v tokens into %v bytes over %v",
		len(*gpt2Encoded), tokenNumBytes, duration,
	)
}

func BenchmarkGPTEncoder_Encode(b *testing.B) {
	start := time.Now()
	tokenCt := len(*gpt2Encoder.Encode(&corpus))
	duration := time.Since(start)
	b.Logf(
		"%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration,
	)
}

func BenchmarkGPTEncoder_EncodeBuffer(b *testing.B) {
	corpusBytes := []byte(corpus)
	start := time.Now()
	_, tokenCt := gpt2Encoder.EncodeBuffer(&corpusBytes)
	duration := time.Since(start)
	b.Logf(
		"%v bytes into %v tokens over %v",
		len(corpus), tokenCt, duration,
	)
}

//func BenchmarkGPTEncoder_WordSplitterTokensChan(b *testing.B) {
//	b.StopTimer()
//	corpusHandle, err := os.Open(largeCorpusPath)
//	//corpusText, err := ioutil.ReadFile(largeCorpusPath)
//	nerdstashEncoder.SplitterThreads = 1
//	//defer corpusHandle.Close()
//	if err != nil {
//		b.Error(err)
//	}
//	wordCount := 0
//	tokensCount := 0
//	runeReader := bufio.NewReaderSize(corpusHandle, 8*1024*1024)
//	wordsChan := make(chan *string, 1000)
//	go nerdstashEncoder.splitWordsOntoChan(runeReader.ReadRune,
//		wordsChan)
//	start := time.Now()
//	b.StartTimer()
//	for {
//		word := <-wordsChan
//		if word == nil {
//			break
//		}
//		tokensCount += len(gpt2Encoder.ToBPE(*word))
//		wordCount++
//	}
//	b.StopTimer()
//	elapsed := time.Since(start)
//	//numBytes := int64(len(corpusText))
//	numBytes, _ := corpusHandle.Seek(0, io.SeekCurrent)
//	b.ReportMetric(float64(wordCount)/elapsed.Seconds(), "words/sec")
//	b.ReportMetric(float64(wordCount), "words")
//	b.ReportMetric(float64(numBytes)/elapsed.Seconds(), "bytes/sec")
//	b.ReportMetric(float64(numBytes), "bytes")
//	b.ReportMetric(float64(tokensCount)/elapsed.Seconds(), "tokens/sec")
//	b.ReportMetric(float64(tokensCount), "tokens")
//}
