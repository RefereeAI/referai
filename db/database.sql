-- PostgreSQL database dump

-- Dumped from database version 17.4
-- Dumped by pg_dump version 17.4

-- Started on 2025-05-20 18:04:31

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
--SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

CREATE TYPE public.foulprediction AS ENUM (
    'NO_FOUL',
    'FOUL'
);

CREATE TYPE public.severityprediction AS ENUM (
    'NO_CARD',
    'RED_CARD',
    'YELLOW_CARD'
);

SET default_tablespace = '';
SET default_table_access_method = heap;

CREATE TABLE public.action (
    id integer NOT NULL,
    user_id integer NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE SEQUENCE public.action_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.action_id_seq OWNED BY public.action.id;

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);

CREATE TABLE public.clip (
    id integer NOT NULL,
    action_id integer NOT NULL,
    content bytea NOT NULL
);

CREATE SEQUENCE public.clip_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.clip_id_seq OWNED BY public.clip.id;

CREATE TABLE public.prediction (
    id integer NOT NULL,
    action_id integer NOT NULL,
    is_foul boolean NOT NULL,
    no_card_confidence double precision NOT NULL,
    red_card_confidence double precision NOT NULL,
    yellow_card_confidence double precision NOT NULL,
    foul_confidence double precision NOT NULL,
    no_foul_confidence double precision NOT NULL,
    foul_model_results json NOT NULL,
    severity_model_results json NOT NULL
);

CREATE SEQUENCE public.prediction_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.prediction_id_seq OWNED BY public.prediction.id;

CREATE TABLE public.user_account (
    id integer NOT NULL,
    email character varying(255) NOT NULL,
    password character varying(255) NOT NULL
);

CREATE SEQUENCE public.user_account_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.user_account_id_seq OWNED BY public.user_account.id;

ALTER TABLE ONLY public.action ALTER COLUMN id SET DEFAULT nextval('public.action_id_seq'::regclass);
ALTER TABLE ONLY public.clip ALTER COLUMN id SET DEFAULT nextval('public.clip_id_seq'::regclass);
ALTER TABLE ONLY public.prediction ALTER COLUMN id SET DEFAULT nextval('public.prediction_id_seq'::regclass);
ALTER TABLE ONLY public.user_account ALTER COLUMN id SET DEFAULT nextval('public.user_account_id_seq'::regclass);

ALTER TABLE ONLY public.action
    ADD CONSTRAINT action_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);

ALTER TABLE ONLY public.clip
    ADD CONSTRAINT clip_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.prediction
    ADD CONSTRAINT prediction_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.user_account
    ADD CONSTRAINT user_account_email_key UNIQUE (email);

ALTER TABLE ONLY public.user_account
    ADD CONSTRAINT user_account_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.action
    ADD CONSTRAINT action_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.user_account(id);

ALTER TABLE ONLY public.clip
    ADD CONSTRAINT clip_action_id_fkey FOREIGN KEY (action_id) REFERENCES public.action(id);

ALTER TABLE ONLY public.prediction
    ADD CONSTRAINT prediction_action_id_fkey FOREIGN KEY (action_id) REFERENCES public.action(id);

-- Completed on 2025-05-20 18:04:31

-- PostgreSQL database dump complete

INSERT INTO public.user_account (email, password)
VALUES ('admin@admin.com', '$2b$12$el9EvM5vq8JUZeRURMCd9uZiyvaaKNvN7jyz1OKRGL0QMgTj4Dk3S');
