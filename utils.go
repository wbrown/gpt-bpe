package gpt_bpe

import (
	"strings"
)

type TrimDirection uint

const (
	TrimTop    TrimDirection = iota
	TrimBottom TrimDirection = iota
	TrimNone   TrimDirection = iota
)

func (encoder *GPTEncoder) TrimNewlines(
	tokens *Tokens,
	direction TrimDirection,
	limit uint,
) (*Tokens, error) {
	var err error
	trimmed := make(Tokens, 0)
	if uint(len(*tokens)) <= limit {
		return tokens, err
	} else if direction == TrimNone {
		return &trimmed, err
	}
	lines := strings.Split(encoder.Decode(tokens), "\n")
	var start, end, step, idx int
	switch direction {
	case TrimTop:
		start = len(lines) - 1
		end = -1
		step = -1
	case TrimBottom:
		start = 0
		end = len(lines)
		step = 1
	}
	accTokens := make(Tokens, 0)
	for idx = start; idx != end; idx += step {
		line := lines[idx]
		switch direction {
		case TrimTop:
			line = "\n" + line
		case TrimBottom:
			line = line + "\n"
		}
		newTokens := encoder.Encode(&line)
		if len(*newTokens)+len(accTokens) > int(limit) {
			return &accTokens, err
		} else {
			switch direction {
			case TrimTop:
				accTokens = append(*newTokens, accTokens...)
			case TrimBottom:
				accTokens = append(accTokens, *newTokens...)
			}
		}
	}
	return &accTokens, err
}

func (encoder *GPTEncoder) AlignAndSizeTokens(
	tokens *Tokens,
	desiredLength int,
) (
	alignedTokens Tokens,
	endAt int,
) {
	if len(*tokens) == desiredLength {
		return *tokens, desiredLength
	}
	var chunk Tokens
	if len(*tokens) < desiredLength {
		chunk = *tokens
		desiredLength = len(*tokens)
	} else {
		chunk = (*tokens)[0:desiredLength]
	}

	// We trim to valid tokens, as we don't want partials
	// that are truncated multi-tokens.
	trimmed := encoder.TrimTokens(&chunk)
	trimmedLength := len(*trimmed)
	isTrimmed := len(*trimmed) != len(chunk)
	chunk = *trimmed
	idx := trimmedLength

	// We do a decode and reencode pass, as this can affect
	// the size after a trim.
	if isTrimmed {
		decodedChunk := encoder.Decode(&chunk)
		reencodedChunk := encoder.Encode(&decodedChunk)
		chunk = *reencodedChunk
		// See if there's any change in size that causes it to
		// be smaller than the `desiredLength`.
		roundtripRemainder := desiredLength - len(chunk)
		if roundtripRemainder > 0 {
			addlEnd := idx + roundtripRemainder
			addlTokens := (*tokens)[idx:addlEnd]
			trimmedAddl := encoder.TrimTokens(&addlTokens)
			chunk = append(chunk, *trimmedAddl...)
			idx += len(*trimmedAddl)
			// Another decode/re-encode pass.
			decodedChunk = encoder.Decode(&chunk)
			reencodedChunk = encoder.Encode(&decodedChunk)
			// Loop, dropping tokens one by one until we have
			// valid tokens and we fit within `contextSize`.
			for {
				chunk = *reencodedChunk
				if len(chunk) <= desiredLength &&
					encoder.TokensReady(&chunk) {
					break
				}
				chunk = chunk[:len(chunk)-1]
				idx -= 1
				decodedChunk = encoder.Decode(&chunk)
				reencodedChunk = encoder.Encode(&decodedChunk)
			}
		}
	}

	return chunk, idx
}
